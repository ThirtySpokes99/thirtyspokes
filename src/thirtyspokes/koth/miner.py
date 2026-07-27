"""The KOTH miner daemon (docs/DESIGN.md §3) — the deployable, decoupled miner.

Unlike the sim (where the validator calls `produce` directly), a real miner runs
independently: once, it publishes its public bundle to its own repo and commits a
salted hash on-chain; each epoch, it derives the epoch nonce from the chain, runs
the owner-given suite inside its TEE runtime (mock only when run offline), and uploads the
attested `proof.json` to its repo. The validator polls the chain and downloads it.

A `reference_agent` baseline artifact is provided: its source is a real single-shot
router that calls one pool model via the metered `call_model` and returns the answer.
Miners replace it with their own routing/orchestration source + weights.
"""

from __future__ import annotations

import time

from ..gateway.gateway import ModelBackend
from ..tee.attestation import Platform
from .benchmarks import Benchmark
from .commit import commit_string, verify_commit
from .epoch import EPOCH_BLOCKS, current_epoch, epoch_nonce
from .runtime import Artifact, KOTHRuntime

# A real baseline agent: single-shot call to one model, return its answer verbatim.
# `weights` is JSON `{"model": "<id>"}`; the source is what the runtime hashes + runs.
#
# THE GENERATION PARAMS MATTER, AND THEY MATCH THE POOL REFERENCE ON PURPOSE. The ranked benchmark is
# code, where an answer is a whole program: the old 512-token cap truncated solutions mid-function
# and graded them as incapability. And `max_tokens` counts THINKING tokens, so an unbounded reasoning
# model spends the entire budget deliberating and returns nothing — measured at 8k and again at 32k.
# `koth/reference.py` measures every pool model at these same settings, so a miner that keeps them is
# compared against a frontier built under identical conditions. A miner is free to change them (that
# is part of the competition), but lowering them below what the reference used means competing
# against models that were given more room than you gave yours.
REFERENCE_SRC = (
    "import json\n"
    "def build_agent(weights):\n"
    "    cfg = json.loads(weights.decode())\n"
    "    model = cfg.get('model', 'mid')\n"
    "    def agent(prompt, call_model):\n"
    "        return call_model(model, [{'role': 'user', 'content': prompt}],\n"
    "                          {'max_tokens': 16384, 'reasoning': {'effort': 'low'}})\n"
    "    return agent\n"
)


def reference_artifact(model: str = "mid") -> Artifact:
    import json
    return Artifact(REFERENCE_SRC, json.dumps({"model": model}).encode(), model)


class KOTHMinerNeuron:
    def __init__(self, hotkey: str, backend: ModelBackend, platform: Platform,
                 suite: list[Benchmark], chain, store, artifact: Artifact, repo: str, *,
                 n_per_bench: int = 8, epoch_blocks: int = EPOCH_BLOCKS, confine: bool = False,
                 commit_proofs: bool = False, confine_timeout: float = 120.0):
        self.hotkey = hotkey                    # ss58 wallet hotkey on a real chain
        self.rt = KOTHRuntime(backend, platform, confine=confine,
                              confine_timeout=confine_timeout)
        self.suite, self.chain, self.store = suite, chain, store
        self.artifact, self.repo = artifact, repo
        self.n_per_bench, self.epoch_blocks = n_per_bench, epoch_blocks
        # F7 anti-grind: also commit the proof's report_data on-chain intra-epoch (before upload), so a
        # commit_window validator can bind us to one run.
        #
        # OFF BY DEFAULT, and gated on the artifact commit having REVEALED (see `_artifact_revealed`).
        # The chain gives each hotkey ONE commitment slot: `CommitmentOf[(netuid, hotkey)]`. Both
        # `set_commitment` (this proof commit) and `set_reveal_commitment` (the artifact commit) write
        # it, so a proof commit issued while the artifact commit is still timelocked OVERWRITES it and
        # the artifact never reveals — validators then bind no artifact and score nobody. Verified
        # on testnet 526: the TimelockEncrypted field was replaced by the plain Raw one 4 blocks later.
        # Once revealed, the record lives in the separate append-only RevealedCommitments map, where a
        # later proof commit cannot touch it — so committing proofs is safe only from that point on.
        self.commit_proofs = commit_proofs
        self._committed = False

    def publish(self) -> None:
        """Upload the public bundle + commit the salted hash once."""
        self.chain.register(self.hotkey)
        # the store returns the IMMUTABLE revision (HF commit SHA) it published at; commit THAT, so
        # the on-chain record pins one snapshot instead of a mutable branch name (it was the literal
        # string "rev1", which the validator parsed and then ignored).
        rev = self.store.upload(self.repo, self.artifact)      # raises if HF gives no SHA
        self.chain.commit(self.hotkey, commit_string(
            self.hotkey, self.repo, rev, self.artifact.source_hash, self.artifact.weights_hash))
        self._committed = True

    def _artifact_revealed(self) -> bool:
        """Has OUR artifact commit made it out of the timelock and into RevealedCommitments?

        Until it has, it sits in the single `CommitmentOf` slot and any proof commit would destroy
        it. Re-publishing a new artifact puts a fresh commit back in that slot, and the hash check
        below stops matching, so proof commits pause again for that reveal window — automatically."""
        try:
            mine = [c for c in self.chain.revealed_commitments() if c.hotkey == self.hotkey]
        except Exception:  # noqa: BLE001 — an unreadable chain is "not revealed yet", never a crash
            return False
        return any(verify_commit(c.data, self.hotkey,
                                 self.artifact.source_hash, self.artifact.weights_hash)
                   for c in mine)

    def run_once(self, epoch: int | None = None) -> int:
        """Produce + upload the attested proof AND its trace for the current epoch."""
        import json
        if not self._committed:
            self.publish()
        if epoch is None:
            epoch = current_epoch(self.chain, self.epoch_blocks)
        nonce = epoch_nonce(epoch, self.chain.beacon(epoch))    # both sides derive the same
        proof, trace = self.rt.run(self.artifact, hotkey=self.hotkey, epoch=epoch, nonce=nonce,
                                   suite=self.suite, n_per_bench=self.n_per_bench)
        if self.commit_proofs and self._artifact_revealed():
            self.chain.commit_proof(self.hotkey, epoch, proof.report_data())  # bind THIS run before revealing
        self.store.upload_proof(self.repo, epoch, proof.to_json())
        self.store.upload_trace(self.repo, epoch, json.dumps(trace))
        return epoch

    def run_forever(self, poll_s: float = 5.0) -> None:  # pragma: no cover — daemon loop
        self.publish()
        last = -1
        while True:
            epoch = current_epoch(self.chain, self.epoch_blocks)
            if epoch != last:
                self.run_once(epoch)
                last = epoch
            time.sleep(poll_s)


def main() -> None:  # pragma: no cover — live daemon (needs bittensor + HF + wallet)
    import argparse

    from ..eval import config
    from ..gateway.gateway import OpenRouterBackend
    from ..subnet.chain import BittensorChain
    from . import tdx
    from .benchmarks import real_suite
    from .pool import PinnedBackend
    from .runtime import mock_vendor_platform
    from .store import HFBundleStore

    p = argparse.ArgumentParser(description="KOTH miner daemon (Bittensor mainnet; runs in your TEE)")
    p.add_argument("--netuid", type=int, required=True)
    p.add_argument("--wallet", required=True, help="Bittensor wallet (coldkey) name")
    p.add_argument("--hotkey", default="default",
                   help="hotkey NAME inside the wallet (the one registered on the subnet). A wallet "
                        "can hold several; only 'default' was reachable before.")
    p.add_argument("--network", default="finney")
    p.add_argument("--repo", required=True, help="your OWN HF model repo, e.g. you/koth-miner")
    p.add_argument("--source", help="path to YOUR build_agent(weights) source (else the reference router)")
    p.add_argument("--weights", help="path to YOUR weights.bin")
    p.add_argument("--model", default="openai/gpt-4o-mini", help="reference router's pool model (if no --source)")
    p.add_argument("--pool", default="openai/gpt-4o-mini,anthropic/claude-opus-4.7",
                   help="owner-pinned model allow-list (comma-separated) — publish this from the subnet owner")
    p.add_argument("--n-per-bench", type=int, default=8)
    p.add_argument("--poll", type=float, default=12.0)
    p.add_argument("--confine", action="store_true",
                   help="run the agent in the no-egress netns confinement (production CVM)")
    p.add_argument("--commit-proofs", action="store_true",
                   help="F7 anti-grind: commit each epoch's proof digest on-chain. Enable ONLY when "
                        "the subnet's validators run --commit-window (otherwise nobody reads it). "
                        "Held back automatically until your artifact commit has revealed, since both "
                        "share one commitment slot and would otherwise destroy it.")
    args = p.parse_args()

    cfg = config.LiveConfig(); cfg.require_key()
    allow = {m.strip() for m in args.pool.split(",")}
    backend = PinnedBackend(OpenRouterBackend(cfg.api_key, cfg.base_url, price_fn=config.price_for), allow)
    if args.source:
        weights = open(args.weights, "rb").read() if args.weights else b'{"model": "%s"}' % args.model.encode()
        artifact = Artifact(open(args.source).read(), weights, "custom")
    else:
        artifact = reference_artifact(args.model)
    # real TDX platform on a confidential VM (auto-detected), else the offline/dev mock vendor key
    platform = tdx.TDXPlatform() if tdx.tdx_available() else mock_vendor_platform()
    if not tdx.tdx_available():
        print("[koth-miner] WARNING: no Intel TDX detected — using the MOCK TEE (zero security). "
              "Mainnet validators REJECT mock proofs (mock_quote_rejected), so you will earn NOTHING. "
              "The mock TEE is for offline/dev only. To mine, boot the owner's published measured "
              "image on a TDX confidential VM so your proof carries a gated hardware quote.")
    chain = BittensorChain(args.netuid, args.wallet, args.network, hotkey=args.hotkey)
    neuron = KOTHMinerNeuron(
        chain.hotkey_ss58(), backend, platform, real_suite(), chain,
        HFBundleStore(), artifact, args.repo, n_per_bench=args.n_per_bench, confine=args.confine,
        commit_proofs=args.commit_proofs)            # F7: opt-in (see --commit-proofs)
    rtmr3 = neuron.rt.measure_self(allow)       # bind runtime -> RTMR3 once at daemon startup (TDX only)
    print(f"[koth-miner] hotkey={neuron.hotkey} repo={args.repo} netuid={args.netuid} "
          f"tdx={tdx.tdx_available()} rtmr3={(rtmr3 or '-')[:16]}")
    neuron.run_forever(poll_s=args.poll)


if __name__ == "__main__":
    main()
