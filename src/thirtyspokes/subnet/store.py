"""Artifact store — where policy weights live (HuggingFace in production).

LocalArtifactStore (a directory) stands in for HuggingFace for offline runs and
tests. HFArtifactStore is the production seam (huggingface_hub upload/download).
"""

from __future__ import annotations

import os

from .artifact import PolicyArtifact


class LocalArtifactStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, repo: str) -> str:
        return os.path.join(self.root, repo.replace("/", "__") + ".npz")

    def upload(self, repo: str, art: PolicyArtifact) -> None:
        art.save(self._path(repo))

    def exists(self, repo: str) -> bool:
        return os.path.exists(self._path(repo))

    def download(self, repo: str) -> PolicyArtifact:
        return PolicyArtifact.load(self._path(repo))


class HFArtifactStore:  # pragma: no cover — production seam
    """huggingface_hub-backed store: upload private, flip public after commit."""

    def __init__(self):
        from huggingface_hub import HfApi  # noqa: PLC0415
        self.api = HfApi()

    def upload(self, repo: str, art: PolicyArtifact) -> None:
        import tempfile

        from huggingface_hub import upload_file
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            art.save(f.name)
            self.api.create_repo(repo, private=True, exist_ok=True, repo_type="model")
            upload_file(path_or_fileobj=f.name, path_in_repo="policy.npz",
                        repo_id=repo, repo_type="model")

    def download(self, repo: str) -> PolicyArtifact:
        from huggingface_hub import hf_hub_download
        return PolicyArtifact.load(hf_hub_download(repo, "policy.npz", repo_type="model"))
