"""Artifact-bundle store — the miner's public HuggingFace repo (docs/DESIGN.md §3).

The bundle a miner publishes to its OWN repo is public and auditable: the routing
source + weights (and, in production, the attested `proof.json`). `hash_source`/
`hash_weights` produce the binding hashes that go into both the attested payload and
the on-chain commit; the validator DOWNLOADS the bundle and recomputes them, so the
public repo, the enclave run, and the chain commitment are provably the same artifact.

`LocalBundleStore` is directory-backed for offline runs; `HFBundleStore` is the production seam. The
repo is PUBLIC (the anti-cheat argument is that anyone can audit the artifact) and `upload` returns the
immutable revision — the HF commit SHA — which the miner puts on-chain and the validator downloads at,
so every validator grades the same snapshot and a later auditor sees exactly the bytes that were scored.
"""

from __future__ import annotations

import os

from ..gateway import signing


def is_pinned_revision(revision) -> bool:
    """A revision must be an immutable commit SHA — never a branch name.

    Branches move. Committing `main` on-chain would mean the bytes a validator grades depend on when
    it happens to look, and an auditor checking the king's code later would see whatever the repo
    holds *now* instead of what was actually scored. So a branch-shaped revision is rejected at both
    ends: the miner refuses to commit one, and the validator refuses to download one."""
    if not isinstance(revision, str):
        return False
    r = revision.strip()
    return len(r) >= 7 and all(c in "0123456789abcdef" for c in r.lower())


def hash_source(source_text: str) -> str:
    return signing.sha256_hex(source_text.encode())


def hash_weights(weights: bytes) -> str:
    return signing.sha256_hex(weights)


class LocalBundleStore:
    """Directory-backed artifact store for offline runs (stands in for HuggingFace)."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _dir(self, repo: str) -> str:
        d = os.path.join(self.root, repo.replace("/", "__"))
        os.makedirs(d, exist_ok=True)
        return d

    def upload(self, repo: str, artifact) -> str:
        d = self._dir(repo)
        with open(os.path.join(d, "source.py"), "w") as f:
            f.write(artifact.source_text)
        with open(os.path.join(d, "weights.bin"), "wb") as f:
            f.write(artifact.weights)
        with open(os.path.join(d, "model_id.txt"), "w") as f:
            f.write(artifact.model_id)
        # stand-in for the HF commit SHA: content-addressed, so it changes iff the bundle changes
        return hash_source(artifact.source_text)[:40]

    def download(self, repo: str, revision: str):
        from .runtime import Artifact
        if not is_pinned_revision(revision):
            raise ValueError(f"refusing to download an unpinned revision: {revision!r}")
        d = self._dir(repo)
        with open(os.path.join(d, "source.py")) as f:
            source_text = f.read()
        with open(os.path.join(d, "weights.bin"), "rb") as f:
            weights = f.read()
        with open(os.path.join(d, "model_id.txt")) as f:
            model_id = f.read()
        return Artifact(source_text, weights, model_id)

    def exists(self, repo: str) -> bool:
        return os.path.exists(os.path.join(self._dir(repo), "source.py"))

    # --- per-epoch attested proof (miner uploads, validator downloads) -------
    def upload_proof(self, repo: str, epoch: int, proof_json: str) -> None:
        pd = os.path.join(self._dir(repo), "proofs")
        os.makedirs(pd, exist_ok=True)
        with open(os.path.join(pd, f"{epoch}.json"), "w") as f:
            f.write(proof_json)

    def download_proof(self, repo: str, epoch: int) -> str | None:
        p = os.path.join(self._dir(repo), "proofs", f"{epoch}.json")
        return open(p).read() if os.path.exists(p) else None

    def upload_trace(self, repo: str, epoch: int, trace_json: str) -> None:
        td = os.path.join(self._dir(repo), "traces")
        os.makedirs(td, exist_ok=True)
        with open(os.path.join(td, f"{epoch}.json"), "w") as f:
            f.write(trace_json)

    def download_trace(self, repo: str, epoch: int) -> str | None:
        p = os.path.join(self._dir(repo), "traces", f"{epoch}.json")
        return open(p).read() if os.path.exists(p) else None

    # --- the public leaderboard feed (validator publishes, dashboard fetches) ---
    def upload_standings(self, repo: str, standings_json: str) -> None:
        with open(os.path.join(self._dir(repo), "standings.json"), "w") as f:
            f.write(standings_json)

    def download_standings(self, repo: str) -> str | None:
        p = os.path.join(self._dir(repo), "standings.json")
        return open(p).read() if os.path.exists(p) else None


class HFBundleStore:  # pragma: no cover — production seam
    """huggingface_hub-backed bundle: routing source + weights (+ proof.json)."""

    def __init__(self, token: str | None = None):
        # The miner and the owner are DIFFERENT HuggingFace identities in production, so the token is
        # explicit rather than a single ambient HF_TOKEN. A validator needs none: miner bundles are
        # public (that is what makes them auditable).
        import os

        from huggingface_hub import HfApi
        self.token = token or os.environ.get("MINER_HF_TOKEN") or os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.token)

    def upload(self, repo: str, artifact) -> str:
        """Publish the bundle PUBLICLY and return the HF commit SHA (the immutable revision).

        The repo must be public: the entire anti-cheat argument is that the artifact is auditable by
        anyone (hardcoded answer tables are visible in the public source). It used to be created
        `private=True`, with the `make_public` helper never called from anywhere — so a validator,
        authenticating as a DIFFERENT HF account, could not read it at all and every miner would have
        been DQ'd `artifact_unavailable`.

        The returned SHA is what goes on-chain, so the commit pins an IMMUTABLE snapshot rather than
        a mutable branch name.
        """
        import tempfile
        from huggingface_hub import upload_file
        self.api.create_repo(repo, private=False, exist_ok=True, repo_type="model")
        self.make_public(repo)                     # idempotent; fixes a repo that already existed private
        rev = None
        for name, data in (("source.py", artifact.source_text.encode()),
                           ("weights.bin", artifact.weights),
                           ("model_id.txt", artifact.model_id.encode())):
            with tempfile.NamedTemporaryFile() as f:
                f.write(data); f.flush()
                info = upload_file(path_or_fileobj=f.name, path_in_repo=name,
                                   repo_id=repo, repo_type="model", token=self.token)
            rev = getattr(info, "oid", None) or rev        # SHA of the last commit = the full bundle
        if not is_pinned_revision(rev):
            # NEVER fall back to a branch name. A branch is mutable, so committing "main" on-chain
            # would re-open the exact hole the revision pin exists to close: the bytes a validator
            # graded would depend on when it looked. If HF didn't hand back a SHA, that is a bug or an
            # API change — fail loudly rather than publish an unpinnable commit.
            raise RuntimeError(
                f"HuggingFace did not return a commit SHA for {repo} (got {rev!r}). Refusing to "
                "commit a mutable branch name on-chain — the artifact binding requires an immutable "
                "revision.")
        return rev

    def download(self, repo: str, revision: str):
        """Fetch the bundle at the EXACT revision the miner committed on-chain.

        `revision` is REQUIRED and must be a pinned SHA — never a branch. Fetching a branch head means
        the bytes a validator grades depend on when it happens to look, and a later auditor sees
        whatever the repo holds *now* rather than what was actually scored.
        """
        from huggingface_hub import hf_hub_download
        from .runtime import Artifact
        if not is_pinned_revision(revision):
            raise ValueError(f"refusing to download an unpinned revision: {revision!r}")
        kw = {"repo_type": "model", "revision": revision}
        src = open(hf_hub_download(repo, "source.py", **kw)).read()
        wts = open(hf_hub_download(repo, "weights.bin", **kw), "rb").read()
        mid = open(hf_hub_download(repo, "model_id.txt", **kw)).read()
        return Artifact(src, wts, mid)

    def exists(self, repo: str) -> bool:
        from huggingface_hub import repo_exists
        return repo_exists(repo, repo_type="model")

    def make_public(self, repo: str) -> None:
        """Flip the repo public after the on-chain commit reveals — the bundle must be
        auditable by anyone (public source + weights is what makes cheating detectable)."""
        self.api.update_repo_settings(repo_id=repo, repo_type="model", private=False)

    def _upload_json(self, repo: str, path_in_repo: str, payload: str) -> None:
        import tempfile

        from huggingface_hub import upload_file
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            f.write(payload); f.flush()
            upload_file(path_or_fileobj=f.name, path_in_repo=path_in_repo,
                        repo_id=repo, repo_type="model", token=self.token)

    def _download_json(self, repo: str, path_in_repo: str) -> str | None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
        try:
            return open(hf_hub_download(repo, path_in_repo, repo_type="model")).read()
        except (EntryNotFoundError, RepositoryNotFoundError):     # not yet uploaded / no repo
            return None

    def upload_proof(self, repo: str, epoch: int, proof_json: str) -> None:
        self._upload_json(repo, f"proofs/{epoch}.json", proof_json)

    def download_proof(self, repo: str, epoch: int) -> str | None:
        return self._download_json(repo, f"proofs/{epoch}.json")

    def upload_trace(self, repo: str, epoch: int, trace_json: str) -> None:
        self._upload_json(repo, f"traces/{epoch}.json", trace_json)

    def download_trace(self, repo: str, epoch: int) -> str | None:
        return self._download_json(repo, f"traces/{epoch}.json")

    # The public leaderboard feed: ONE json at a stable path the dashboard fetches. Overwritten each
    # settled epoch — it is a snapshot of the standings, not an append-only log.
    def upload_standings(self, repo: str, standings_json: str) -> None:
        # The repo must EXIST and be PUBLIC — the dashboard fetches it with no token. `_upload_json`
        # never created it (only the miner-bundle `upload` did), so a validator's very first publish
        # 404'd, `flush_standings` swallowed it (it must never break scoring), and the dashboard
        # showed "no validator feed" forever with nothing in the logs. Seen live on testnet 526.
        self.api.create_repo(repo, private=False, exist_ok=True, repo_type="model")
        # `create_repo(..., private=False, exist_ok=True)` does not change an existing private repo.
        # The browser fetches anonymously, so repair that state before every publish.
        self.make_public(repo)
        self._upload_json(repo, "standings.json", standings_json)

    def download_standings(self, repo: str) -> str | None:
        return self._download_json(repo, "standings.json")
