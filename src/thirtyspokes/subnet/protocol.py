"""Protocol — the fixed rules of the game (ARCHITECTURE.md §3 "protocol-owns").

Single source of truth for what every miner and validator must agree on: the
sanctioned pool, the orchestration engine version + I/O contract, the artifact
size caps, the cost-aware objective weight, the reign parameters, and the
commit-reveal format. Miners submit a policy that plugs into THIS; they never
change it. Bumping a version here starts a new competition season.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..orchestrator import N_ROLES, N_STATE
from ..synth_tinyrouter import TinyConfig

# --- versions (bump to start a new season) ----------------------------------
NETUID = 0                      # placeholder; set for testnet/mainnet
ENGINE_VERSION = "orchestra-engine-1"   # the fixed multi-turn role loop
CONTRACT_VERSION = "io-contract-1"      # state features -> action space
COMMIT_PREFIX = "orch1"         # on-chain reveal prefix: orch1|<repo>|<sha256>

# --- policy artifact limits (the I/O contract is fixed; architecture is free) -
MAX_POLICY_PARAMS = 5_000_000   # cap on the miner's policy head
ALLOWED_HIDDEN = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256)  # customizable capacity

# --- sanity gate thresholds --------------------------------------------------
MIN_ACTION_ENTROPY = 0.15       # reject degenerate constant policies
PROBE_SIZE = 256                # probe set size for fingerprint + sanity

# --- fingerprint dedup -------------------------------------------------------
DUP_TV_THRESHOLD = 0.03         # calibrate in Phase-0 vs independent policies


@dataclass(frozen=True)
class CompetitionParams:
    """Everything an epoch is scored under. Frozen within a season."""

    lam: float = 0.15                    # cost weight in quality - lam*cost
    max_turns: int = 5                   # orchestration budget (turns)
    eval_questions: int = 4000           # tasks sampled per epoch
    # reign
    reign_split: tuple[float, ...] = (0.40, 0.25, 0.15, 0.12, 0.08)
    eps0: float = 0.02
    eps_floor: float = 0.002
    eps_tau: float = 8.0
    # the sanctioned pool (owner-curated). In production this is a live-model
    # gateway spec; here it is the synthetic pool definition.
    pool: TinyConfig = field(default_factory=lambda: TinyConfig(q=4000, seed=7))

    @property
    def n_roles(self) -> int:
        return N_ROLES

    @property
    def n_state(self) -> int:
        return N_STATE
