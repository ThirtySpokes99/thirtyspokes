"""The dev kit's promise (the build plan M6b, docs/WHITEPAPER.md — the "Kind" property).

    what I see locally is what the validator scores.

A miner gets one submission per hotkey and an invalid artifact spends it, so every test here is a
test of that sentence rather than of a convenience. The first one is the load-bearing one: the kit
must call the validator's own functions, because a reimplementation drifts and it drifts precisely
for the miners who most need it. The rest pin the three things a miner cannot otherwise discover
without burning a registration — that their model would be refused, that its turns do not parse, and
that its episodes never reach STOP — and that none of it needs a credential or a socket.
"""

from __future__ import annotations

import json
import shutil
import socket
from dataclasses import replace
from pathlib import Path

from unittest import mock

import pytest

from thirtyspokes.v3 import admission, devkit, scaffold, score
from thirtyspokes.v3 import reference as reference_module   # `reference` is a fixture below
from thirtyspokes.v3.admission import AdmissionError, admit, describe
from thirtyspokes.v3.benchmarks.base import inspector
from thirtyspokes.v3.benchmarks.mock import MockBenchmark, attempt, in_process
from thirtyspokes.v3.config import (
    DUEL_WALL_CLOCK_REASON,
    EPISODE_CONCURRENCY,
    EPISODE_WALL_CLOCK_SECONDS,
    MAX_CONSECUTIVE_PARSE_FAILURES,
    MAX_STEPS,
)
from thirtyspokes.v3.devkit import (
    ENDINGS,
    check_admission,
    dry_run,
    format_report,
    main,
    mock_world,
    suite_inspect,
)
from thirtyspokes.v3.reference import FixedPolicy, always_cheapest, always_strongest, cascade
from thirtyspokes.v3.scaffold import Scaffold
from thirtyspokes.v3.score import score_arm
from thirtyspokes.v3.types import TOOL_NAMES, ToolCall


@pytest.fixture
def world():
    return mock_world()


def two_rung_cascade(world):
    """Cheap first, the top rung on failure — the free win a Conductor has to beat (§5.5)."""
    return cascade(world.catalog, rungs=2)


class Gibberish:
    """A model whose every turn is refused — §8b.2's pathological case, and the one a miner is most
    likely to ship by accident: a fine-tune that answers the task instead of choosing who will."""

    def act(self, prompt: str) -> str:
        return "I think the answer is 42."


class NeverStops:
    """Architecturally legal, behaviourally useless: it parses fine and never decides to stop."""

    def act(self, prompt: str) -> str:
        return "DELEGATE mock/cheap"


def gibberish_conductor() -> Gibberish:
    """A `--conductor module:attr` factory, in the shape a miner writes one."""
    return Gibberish()


# --- the model tree, in miniature -----------------------------------------------------------------
# Data-driven checks (`name -> (dtype, shape)`, no loaded model) are what make the whole §1.1 gate
# exercisable without the 70 GB artifact it is defined against.
_CONFIG = {"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"],
           "num_hidden_layers": 48, "hidden_size": 4096, "num_experts": 128, "vocab_size": 151936}


def _tree(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(_CONFIG, indent=2))
    (root / "tokenizer.json").write_text('{"model": {"vocab": {"hello": 0}}}')
    (root / "tokenizer_config.json").write_text('{"chat_template": "{{ messages }}"}')
    header = json.dumps({"model.embed_tokens.weight": {"dtype": "BF16", "shape": [32, 8],
                                                       "data_offsets": [0, 512]}}).encode()
    (root / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + bytes(512))
    return root


@pytest.fixture
def reference_tree(tmp_path: Path) -> Path:
    return _tree(tmp_path / "reference")


@pytest.fixture
def reference(reference_tree: Path):
    return describe(reference_tree)


def _copy(reference_tree: Path) -> Path:
    candidate = reference_tree.parent / "candidate"
    shutil.copytree(reference_tree, candidate)
    return candidate


# --- exit 1: the same code, not the same idea -----------------------------------------------------
def test_the_dev_kit_calls_the_validator_own_functions(world):
    """M6b exit 1, and the reason every other test here means anything.

    Identity, not behaviour: a kit that merely *agrees* today with `admit`, `Scaffold` and
    `score_arm` is a kit that can silently stop agreeing after the next change to any of them — and
    the divergence would surface as a miner's shot spent on a surprise. The scoring path is asserted
    at the object level too, since a copied `score_arm` would look right for exactly as long as
    λ_b's definition holds still.
    """
    assert devkit.admit is admission.admit
    assert devkit.Scaffold is scaffold.Scaffold
    assert devkit.score_arm is score.score_arm
    # λ_b's PAIR is a decision, and the kit must take it from the same function the validator does:
    # a kit that fitted its own pair would price a miner's local run on a different line.
    assert devkit.fit_pool is reference_module.fit_pool

    report = dry_run(world, two_rung_cascade(world), budget_usd=100.0)
    assert isinstance(report.arm, score.ArmScore)


# --- exit 2: the identical refusal ----------------------------------------------------------------
def test_a_model_that_would_fail_admission_fails_locally_with_the_identical_string(
        reference_tree, reference):
    """M6b exit 2. The refusal is what a miner greps for and what the operator reads in the
    validator's log; a local paraphrase would be a divergence dressed as a nicety, and the miner
    would discover it after spending their one submission rather than before.

    Three refusals, one per §1.1 property — the pickle archive (no code execution), a changed
    architectural field (architecture identity), and a tokenizer byte (comparability) — because a
    kit that forwarded one message verbatim and reworded the others would pass a single-case test.
    """
    def mutate_pickle(root: Path) -> None:
        (root / "pytorch_model.bin").write_bytes(b"\x80\x04")

    def mutate_config(root: Path) -> None:
        (root / "config.json").write_text(json.dumps({**_CONFIG, "num_hidden_layers": 47}))

    def mutate_tokenizer(root: Path) -> None:
        (root / "tokenizer.json").write_text('{"model": {"vocab": {"hello": 1}}}')

    for mutate in (mutate_pickle, mutate_config, mutate_tokenizer):
        candidate = _copy(reference_tree)
        mutate(candidate)

        with pytest.raises(AdmissionError) as caught:
            admit(candidate, reference)
        assert check_admission(candidate, reference) == str(caught.value) != ""
        shutil.rmtree(candidate)


def test_a_tree_the_validator_would_admit_reports_no_refusal(reference_tree, reference):
    """The other direction, and the one a false positive would be cruellest in: a kit that refused a
    legal tree would send a miner off to fix a model that was never broken."""
    assert check_admission(reference_tree, reference) is None
    assert check_admission(_copy(reference_tree), reference) is None


# --- exit 3: the same numbers ---------------------------------------------------------------------
def test_the_local_score_is_the_validator_path_bit_for_bit(world):
    """M6b exit 3. Written out as the validator writes it — `Scaffold.run_window` then `score_arm` —
    and compared with `==` rather than with a tolerance on purpose: the kit runs the same functions
    over the same inputs, so the two must agree to the last bit. The day this needs `approx` is the
    day a reimplementation has crept in, and that is what the assertion is watching for."""
    arm = Scaffold(world.catalog, two_rung_cascade(world), world.worker, world.grade)
    episodes = arm.run_window(world.tasks, nonce="w7", budget_usd=100.0)
    expected = score_arm([e.result for e in episodes], world.exchange)

    report = dry_run(world, two_rung_cascade(world), budget_usd=100.0, nonce="w7")
    assert report.arm == expected
    assert report.arm.final == expected.final
    assert [b.benchmark for b in report.arm.per_benchmark] == ["cybergym", "terminal-bench"]
    assert report.spend_usd == pytest.approx(sum(e.result.spend_usd for e in episodes))


def test_the_stand_in_world_has_the_property_the_score_is_built_to_have(world):
    """§5.1b/D8: with λ_b fitted to the pool's own slope, the floor and King₀ score EXACTLY the
    same, so buying quality at the pool's rate earns nothing and only a policy above that line
    scores above both. King₀ in the stand-in world is the three-rung cascade (`fit_pool`), so the
    always-strongest arm — the cascade's quality at top-rung prices — now sits BELOW the line. The
    kit derives its stand-in λ through the validator's own `fit_pool` precisely so a miner's first
    local run teaches that and not an artefact of a guessed constant."""
    cheap = dry_run(world, always_cheapest(world.catalog), budget_usd=100.0)
    king = dry_run(world, cascade(world.catalog), budget_usd=100.0)
    strong = dry_run(world, always_strongest(world.catalog), budget_usd=100.0)

    assert cheap.arm.quality < king.arm.quality
    assert cheap.spend_usd < king.spend_usd
    assert cheap.arm.final == pytest.approx(king.arm.final, abs=1e-12)
    assert strong.arm.final < cheap.arm.final, "overpaying for King0's quality must cost something"

    # A band a sequential policy can capture: skip the overpriced top rung and stop on the
    # good-value one. Without this a miner's first run would teach that routing is pointless.
    skipping = dry_run(world, FixedPolicy("skip-the-apex", ("mock/cheap", "mock/mid")),
                       budget_usd=100.0)
    assert skipping.arm.final > cheap.arm.final


def test_lambda_is_fitted_per_benchmark_not_globally(world):
    """§5.1b: the accuracy-per-dollar exchange rate genuinely differs across benchmarks, so a
    stand-in world whose two benchmarks shared one slope would demonstrate a global λ just as well
    and would let a per-benchmark bug through unnoticed."""
    rates = {b: e.lam for b, e in world.exchange.items()}
    assert set(rates) == {"cybergym", "terminal-bench"}
    assert rates["cybergym"] > rates["terminal-bench"] * 2
    assert all(e.flags == () for e in world.exchange.values()), (
        "a guarded λ is clamped to 0, which would make the stand-in world score pure quality")


# --- exit 4: the two failure modes a score cannot show --------------------------------------------
def test_a_model_whose_turns_never_parse_is_reported_as_such(world):
    """M6b exit 4. This is the failure a fine-tune ships by accident — a model that answers the task
    instead of choosing who will — and in the score it is indistinguishable from bad routing: both
    are a low number. The rate and the ending are what separate them."""
    report = dry_run(world, Gibberish(), budget_usd=100.0)

    assert report.parse_failure_rate == 1.0
    assert report.endings["parse_failures"] == len(world.tasks)
    assert report.endings["stop"] == 0
    assert report.arm.final <= 0.0
    assert report.spend_usd == 0.0, "a model that never parses never reaches a worker"
    turns = MAX_CONSECUTIVE_PARSE_FAILURES * len(world.tasks)
    assert f"parse failures: {turns} of {turns} turns (100.0%)" in format_report(report)


def test_a_recovered_parse_failure_shows_up_as_a_rate_not_as_a_verdict(world):
    """Consecutive, not cumulative (§2) — so a policy that fumbles and recovers still scores, and
    the miner still sees the fumble. A kit that reported only the endings would show this model as
    perfectly healthy."""
    class FumblesOnce:
        """Refused on its first turn, delegates on its second, stops once a call has come back."""

        def act(self, prompt: str) -> str:
            if "(no steps yet)" in prompt:
                return "I think the answer is 42."
            return "STOP" if "->" in prompt else "DELEGATE mock/cheap"

    report = dry_run(world, FumblesOnce(), budget_usd=100.0)

    assert report.endings["stop"] == len(world.tasks)
    assert report.parse_failures == len(world.tasks)
    assert report.parse_failure_rate == pytest.approx(1 / 3)
    assert report.arm.quality > 0.0


def test_step_cap_exhaustion_is_reported(world):
    """M6b exit 4, §8b.2. A model that never reaches STOP burns eleven worker calls on every task —
    the miner's own allowance locally, the king's in a duel — and its score alone would just look
    like an expensive policy."""
    report = dry_run(world, NeverStops(), budget_usd=100.0)

    assert report.endings == {**dict.fromkeys(ENDINGS, 0), "max_steps": len(world.tasks)}
    assert report.turns == MAX_STEPS * len(world.tasks)
    assert report.parse_failure_rate == 0.0, "it parses perfectly; that is the point"
    assert f"max_steps={len(world.tasks)}" in format_report(report)


def test_both_wall_clocks_are_reported_and_are_never_the_same_row(world):
    """M6b exit 4, §8b.2's two bounds — and they must stay distinguishable in the local report.

    Driven through the same `clock` seam the scaffold exposes, so a fifteen-minute timeout is
    visible in microseconds. A clock advancing a full episode's worth per read stalls the first
    episodes into `wall_clock` and then puts the ARM past `DUEL_WALL_CLOCK_SECONDS`, so the tail is
    never reached at all: `DUEL_WALL_CLOCK_SECONDS / EPISODE_WALL_CLOCK_SECONDS` episodes can stall
    before the arm's own bound bites. "Every episode times out" is therefore arithmetically
    impossible for a world of more than that many tasks, which is the denial of service the per-duel
    clock exists to refuse, seen from the miner's side.

    Both endings are asserted separately, never their sum: one says the model stalled and the other
    says the validator ran out of time, and a report that merged them would tell a miner to fix the
    wrong thing (§5.8). The split itself is asserted as a SUFFIX rather than as a count — the arm's
    deadline never un-passes, so every task after the first abandoned one is abandoned too, and how
    many stalled episodes fit inside two hours depends on how often the scaffold reads its clock,
    which is not a property this test is about."""
    ticks = iter(range(1, 10_000))
    report = dry_run(world, two_rung_cascade(world), budget_usd=100.0,
                     clock=lambda: next(ticks) * EPISODE_WALL_CLOCK_SECONDS)

    stalled = report.endings["wall_clock"]
    unreached = report.endings[DUEL_WALL_CLOCK_REASON]
    assert stalled > 0 and unreached > 0, "only one clock fired; this proves nothing about the pair"
    assert stalled + unreached == len(world.tasks)
    assert [e.result.stopped_reason for e in report.episodes] == \
        ["wall_clock"] * stalled + [DUEL_WALL_CLOCK_REASON] * unreached
    # The tail was never reached, so it carries no steps at all — the same distinction §5.8 draws
    # for the two flavours of budget exhaustion.
    assert not any(e.result.steps for e in report.episodes[stalled:])
    assert report.arm.quality == 0.0
    assert report.parse_failure_rate == 0.0, "no turns ran, so no turn was refused"


def test_a_worlds_read_channel_is_the_one_the_validator_runs(world, tmp_path):
    """The "Kind" property applied to the read channel (§2.1, `v3/tools.py`): `LocalWorld.inspect`
    sits beside `grade` because it is the other half of what the ADAPTERS contribute to an episode,
    and `dry_run` passes it to `Scaffold` exactly as `validator._arm` does. Without this the wiring
    would be reachable only in the validator, so a miner's local run and the scored run would differ
    on the one thing the channel changes — what the worker was shown."""
    root = tmp_path / "testbed"
    root.mkdir()
    (root / "mod.py").write_text("def broken():\n    return 1\n")
    bench = MockBenchmark("terminal-bench", {"mock/cheap": 9}, n_tasks=1, tool_names=TOOL_NAMES,
                          root=str(root))

    class Reader:
        """Reads once, then answers — the shape the channel exists to make possible."""

        def __init__(self):
            self.seen = []

        def complete(self, model_id, task_text, params):
            self.seen.append(task_text)
            return ("READ mod.py" if len(self.seen) == 1 else attempt(model_id)), 0.01

    worker = Reader()
    local = replace(world, tasks=bench.load(), worker=worker,
                    inspect=inspector(bench, execute=in_process))

    report = dry_run(local, always_cheapest(local.catalog), budget_usd=100.0)

    assert len(worker.seen) == 2, "the read loop never ran, so the wiring is untested"
    assert worker.seen[0] == local.tasks[0].prompt        # compose(p, ()) == p, byte for byte
    assert "     1\tdef broken():" in worker.seen[1]
    assert report.spend_usd == pytest.approx(0.02), "both turns are the delegate's own cost"


def test_a_multi_benchmark_slice_reaches_each_adapters_own_read_channel(world, tmp_path):
    """`grade` has a corpus dispatcher and `inspect` needs the same one, because a window slice is
    multi-benchmark by construction (§6.3b) while `LocalWorld.inspect` is ONE callable carried into
    every Scaffold. A bare `inspector(one_benchmark)` is correct only while exactly one adapter
    declares tools and `Scaffold._read` short-circuits on every other task's empty `task.tools`; the
    second adapter to declare a verb has its tasks read against the FIRST adapter's environment,
    which is silently the wrong bytes rather than a crash — `MockBenchmark.environment` and
    `r2egym`'s both answer from the benchmark, not from the task. Two roots, so the wrong dispatch
    is visible in what the worker was shown."""
    benches = []
    for name in ("terminal-bench", "cybergym"):       # both are in the stand-in world's λ table
        root = tmp_path / name
        root.mkdir()
        (root / "mod.py").write_text(f"# {name}\n")
        benches.append(MockBenchmark(name, {"mock/cheap": 9}, n_tasks=1, tool_names=TOOL_NAMES,
                                     root=str(root)))

    class Reader:
        """Reads once per delegate, then answers."""

        def __init__(self):
            self.seen = []

        def complete(self, model_id, task_text, params):
            self.seen.append(task_text)
            if "INSPECTION RESULTS" in task_text:
                return attempt(model_id), 0.01
            return "READ mod.py", 0.01

    worker = Reader()
    local = replace(world, tasks=tuple(b.load()[0] for b in benches), worker=worker,
                    inspect=suite_inspect(benches, execute=in_process))

    dry_run(local, always_cheapest(local.catalog), budget_usd=100.0)

    composed = [text for text in worker.seen if "INSPECTION RESULTS" in text]
    assert len(composed) == 2, "both tasks must have reached the read channel"
    for text in composed:
        # `[<benchmark>] task 0: ...` is the prompt; the file's contents say which tree was read.
        # Undispatched, both tasks would have read the FIRST benchmark's, and nothing would raise.
        benchmark = text[1:text.index("]")]
        assert f"# {benchmark}\n" in text, "a task must be read against its own benchmark"


def test_a_read_for_a_tag_no_benchmark_claims_raises_rather_than_reading_nothing(tmp_path):
    """`suite_grade`'s rule at the other seam, and here the alternative to raising is worse than a
    zero: `Scaffold._read` reads None as "the reply is the submission", so a slice built wrong would
    have a worker's `READ` line graded as its patch. A raise instead reaches `_GraderGuard`, which
    is §8b.3's one place that can act on it."""
    root = tmp_path / "testbed"
    root.mkdir()
    bench = MockBenchmark("terminal-bench", {"mock/cheap": 1}, n_tasks=1, tool_names=TOOL_NAMES,
                          root=str(root))
    stranger = replace(bench.load()[0], benchmark="cybergym")

    with pytest.raises(KeyError, match="cybergym"):
        suite_inspect((bench,), execute=in_process)(stranger, ToolCall("READ", "mod.py"), "sha")


def test_assembling_the_stand_in_world_runs_the_channel_check_over_its_own_benchmarks(monkeypatch):
    """`mock_world` is the TEMPLATE a miner copies for their `--world`, so the gate has to be IN it.

    A world whose benchmarks happen to agree passes with the check and without it, which is exactly
    why its absence would be invisible: every assertion about the shipped stand-in stays green while
    the line an owner copies quietly stops carrying the check. So this asserts the call itself, and
    that it is asked about this world's OWN benchmarks rather than about nothing — the same wiring
    the validator's window open is tested for, at the other end of the corpus's life."""
    asked = []
    monkeypatch.setattr(devkit, "check_channel", lambda benchmarks: asked.append(tuple(benchmarks)))

    world = mock_world()

    assert asked, "mock_world assembled a corpus without checking the read channel"
    assert asked[0] == devkit._MOCK_BENCHMARKS
    assert {t.benchmark for t in world.tasks} == {b.name for b in asked[0]}


def test_a_healthy_run_still_names_the_per_duel_clock_at_zero(world):
    """The ending vocabulary must cover every reason `Scaffold` can produce, or the kit shows a
    miner a row its own report has no name for — which is local behaviour diverging from scored
    behaviour, the one thing M6b exists to prevent. `dry_run` calls `Scaffold.run_window`, so the
    per-duel clock is enforced locally exactly as it is when the window is scored."""
    report = dry_run(world, two_rung_cascade(world), budget_usd=100.0)

    assert DUEL_WALL_CLOCK_REASON in ENDINGS
    assert report.endings[DUEL_WALL_CLOCK_REASON] == 0
    assert f"{DUEL_WALL_CLOCK_REASON}=0" in format_report(report)


def test_budget_exhaustion_is_reported_and_distinguishes_the_tail(world):
    """§4 and §5.8. A run decided by funding rather than by routing must be legible as such — the
    tail the allowance never reached carries no steps at all, and the report says so rather than
    leaving a miner to read nine zeroes as a routing failure.

    The two flavours of exhaustion have to stay distinguishable: one episode runs out mid-flight and
    KEEPS the credit it bought, while the tail behind it never spent a Conductor call."""
    report = dry_run(world, always_cheapest(world.catalog), budget_usd=0.035)

    assert report.endings["stop"] == 3
    assert report.endings["budget_exhausted"] == len(world.tasks) - 3
    ran_out = [e for e in report.episodes
               if e.result.stopped_reason == "budget_exhausted" and e.result.steps]
    assert len(ran_out) == 1 and ran_out[0].result.spend_usd == 0.01
    assert "(the allowance never reached this task — §5.8)" in format_report(report)


def test_every_ending_is_named_even_at_zero(world):
    """A healthy run must still print the endings that did not happen. Reporting "this never
    occurred" by absence reads identically to a report that forgot to count it, which is the whole
    failure this section exists to close (the same reason `emissions` always emits the burn key)."""
    report = dry_run(world, two_rung_cascade(world), budget_usd=100.0)

    assert set(report.endings) == set(ENDINGS)
    assert report.endings["stop"] == len(world.tasks)
    for reason in ENDINGS:
        assert f"{reason}=" in format_report(report)


# --- the decisions --------------------------------------------------------------------------------
def test_the_report_names_the_model_at_every_step_and_why_each_episode_ended(world):
    """The kit's other half: which model at which step. A miner tuning a router needs to see WHERE
    the escalation happened, not only that the total was high — and the trajectory is what they
    train on (D15)."""
    text = format_report(dry_run(world, two_rung_cascade(world), budget_usd=100.0))

    assert "step 1: DELEGATE mock/cheap -> failed ($0.010000)" in text
    assert "step 2: RETRY mock/strong -> succeeded ($0.200000)" in text
    assert "step 3: STOP" in text
    assert "cybergym-0003 [cybergym] score 1.000 $0.210000 ended=stop" in text
    assert "step 1: DELEGATE mock/cheap -> succeeded ($0.010000)" in text, (
        "an easy task must show the cheap model succeeding, or escalation reads as unconditional")


def test_the_report_shows_per_benchmark_quality_spend_and_final(world):
    """M6b's reporting requirement, and §5.1's equal weight per BENCHMARK: a miner who only saw a
    total could not tell a broad policy from one carried by a single benchmark — which is exactly
    what the leave-one-out rule will refuse them for (§5.2)."""
    report = dry_run(world, two_rung_cascade(world), budget_usd=100.0)
    text = format_report(report)

    for row in report.arm.per_benchmark:
        assert row.benchmark in text
        assert f"{row.quality:.4f}" in text
        assert f"{row.final:.4f}" in text
    assert f"{report.arm.final:.4f}" in text
    assert report.arm.quality == pytest.approx(
        sum(r.quality for r in report.arm.per_benchmark) / len(report.arm.per_benchmark))


# --- exit 5: no credentials, no chain, no socket --------------------------------------------------
def test_it_runs_with_no_credentials_and_no_network(monkeypatch, capsys):
    """M6b exit 5. A miner must be able to iterate BEFORE registering anything, so the default path
    must not touch R2, the chain or a socket. Asserted by removing all three rather than by reading
    the imports: every credential in the environment is cleared and `socket` is made to raise, so
    any attempt to reach the network fails the test instead of quietly succeeding on the machine
    that happens to have a key."""
    for name in ("OPENROUTER_API_KEY", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
                 "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "BT_WALLET_NAME"):
        monkeypatch.delenv(name, raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("the dev kit opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    main([])

    out = capsys.readouterr().out
    assert "# SCORE" in out and "# FAILURE MODES" in out and "# DECISIONS" in out
    assert "stand-in reference policy 'always-cheapest'" in out, (
        "the default run must say it scored a stand-in, or its number reads as the miner's own")


def test_the_cli_prints_the_validator_refusal_and_refuses_to_print_a_score(
        reference_tree, reference, capsys):
    """A refused artifact is never scored, so a kit that printed a score under a refusal would show
    a miner a number the validator will never compute for them. It exits non-zero so a miner can
    gate on it before uploading 70 GB."""
    candidate = _copy(reference_tree)
    (candidate / "adapter.pt").write_bytes(b"\x80\x04")

    with pytest.raises(SystemExit) as caught:
        main(["--model-tree", str(candidate), "--reference", str(reference_tree)])

    out = capsys.readouterr().out
    assert caught.value.code == 1
    assert check_admission(candidate, reference) in out
    assert "# SCORE" not in out


def test_the_cli_scores_after_an_admitted_tree(reference_tree, capsys):
    """The happy path in one command: admitted, then the report."""
    main(["--model-tree", str(reference_tree), "--reference", str(reference_tree),
          "--budget-usd", "100"])

    out = capsys.readouterr().out
    assert "ADMISSION: admitted" in out
    assert "# SCORE" in out


def test_the_cli_refuses_a_model_tree_with_nothing_to_compare_it_to(reference_tree):
    """There is no default reference: admission is a comparison against the owner's PINNED
    architecture, and a kit that quietly checked a tree against itself would admit anything."""
    with pytest.raises(SystemExit, match="needs --reference"):
        main(["--model-tree", str(reference_tree)])


def test_the_cli_scores_a_supplied_conductor_against_a_supplied_world(capsys):
    """The path a miner actually runs. Both seams are `module:attr` factories, because the two
    things the kit cannot ship are the two things it exists to score: the miner's own weights, and
    the benchmarks with their graders. A kit whose plug-in points were untested would work for its
    author and nobody else."""
    main(["--world", "thirtyspokes.v3.devkit:mock_world",
          "--conductor", "test_devkit:gibberish_conductor", "--budget-usd", "100"])

    out = capsys.readouterr().out
    assert "(100.0%)" in out, "the supplied Conductor's own failure mode must reach the report"
    assert "stand-in" not in out, "a supplied Conductor must not be silently replaced"


def test_a_malformed_plug_in_spec_is_refused_rather_than_guessed(capsys):
    """`module:attr`, not a module a caller hoped had one obvious entry point. Guessing would run
    whichever callable happened to be there — arbitrary code the miner did not name."""
    with pytest.raises(SystemExit, match="module:attr"):
        main(["--conductor", "thirtyspokes.v3.devkit"])


def test_the_kit_runs_a_slice_the_way_the_validator_runs_it(world):
    """The kit's whole promise is that a local run and a scored run agree, and concurrency is part
    of that agreement rather than a speed knob.

    `Validator._arm` runs `EPISODE_CONCURRENCY` episodes at once, which is what lets ~250 tasks
    finish inside the per-duel clock. A kit left at 1 takes hours on the same slice and ends in
    `duel_wall_clock` — showing a miner a failure the validator would never have produced, against
    which they would then tune.
    """
    calls = {}

    real = Scaffold.run_window

    def spy(self, tasks, *, nonce, budget_usd, on_episode=None, concurrency=1):
        calls["concurrency"] = concurrency
        return real(self, tasks, nonce=nonce, budget_usd=budget_usd,
                    on_episode=on_episode, concurrency=concurrency)

    with mock.patch.object(Scaffold, "run_window", spy):
        dry_run(world, two_rung_cascade(world), budget_usd=100.0)

    assert calls["concurrency"] == EPISODE_CONCURRENCY


def test_a_miner_can_still_pin_it_serial_to_read_one_episode_at_a_time(world):
    calls = {}
    real = Scaffold.run_window

    def spy(self, tasks, *, nonce, budget_usd, on_episode=None, concurrency=1):
        calls["concurrency"] = concurrency
        return real(self, tasks, nonce=nonce, budget_usd=budget_usd,
                    on_episode=on_episode, concurrency=concurrency)

    with mock.patch.object(Scaffold, "run_window", spy):
        report = dry_run(world, two_rung_cascade(world), budget_usd=100.0, concurrency=1)

    assert calls["concurrency"] == 1
    assert report.episodes, "a serial run still produces a report"
