"""Startup preflight: the deployment-shape checks that broke live runs (koth/doctor.py)."""

import pathlib

import pytest


def test_preflight_refuses_to_start_a_deployment_with_a_blocking_fault():
    """Both daemons call `preflight_or_exit`. The point of the preflight is that a broken deployment
    never comes up — it used to come up looking healthy and score nothing for hours."""
    from thirtyspokes.koth import doctor

    bad = [doctor._r(doctor.OK, "slice agreement", "all 2"),
           doctor._r(doctor.FAIL, "code grading", "no `docker` client on PATH")]
    with pytest.raises(SystemExit) as e:
        doctor.preflight_or_exit("koth-validator", bad)
    assert "docker" in str(e.value) and "koth-validator" in str(e.value)

    # dev/offline bring-up reports the same thing but proceeds: there is no published governance and
    # often no docker daemon there, and refusing would break the offline sim path
    doctor.preflight_or_exit("koth-validator", bad, block=False)

    ok = [doctor._r(doctor.OK, "slice agreement", "all 2"),
          doctor._r(doctor.WARN, "governance", "skipped")]
    doctor.preflight_or_exit("koth-validator", ok)      # warnings never block


def test_both_daemons_run_the_preflight_before_the_long_loop():
    """A regression here is invisible until a live run wastes hours, so pin it in the source."""
    import inspect

    from thirtyspokes.koth import miner, neuron
    for mod, role in ((neuron, "koth-validator"), (miner, "koth-miner")):
        src = inspect.getsource(mod)
        assert "preflight_or_exit" in src, f"{role} lost its startup preflight"
        assert "--skip-preflight" in src, f"{role} lost its preflight escape hatch"


def test_operator_echoes_heartbeat_lines_as_the_console_actually_formats_them():
    """The enclave's stdout reaches the serial console through the kernel, which prefixes every line
    with a timestamp and the emitting process. A prefix match therefore echoes NOTHING while the
    guest is printing perfectly — measured on the v23 capture boot, which emitted all six lines and
    had none of them surfaced."""
    import inspect

    from thirtyspokes.koth import gcp_operator

    real = ("[    5.113000] systemd[1]: Started koth-runtime.service.\n"
            "[   36.966582] python[427]: KOTH-MODE routing harness=koth-harness-2 rungs=7\n"
            "[   87.957190] python[427]: KOTH-TASK 1/6 rung=1 rungs_used=1 cost=0.00040\n"
            "[  148.759964] python[427]: KOTH-TASK 6/6 rung=4 rungs_used=1 cost=0.00167\n")
    ticks = [ln for ln in real.splitlines() if "KOTH-TASK " in ln]
    assert len(ticks) == 2, "the console format must still be matched by the operator's filter"
    assert "KOTH-TASK " in inspect.getsource(gcp_operator._boot_once)
    assert 'startswith("KOTH-TASK' not in inspect.getsource(gcp_operator._boot_once)


def _publish_script():
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "publish_runtime_image.py"
    spec = importlib.util.spec_from_file_location("publish_runtime_image", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_publish_refuses_when_head_is_not_the_commit_that_built_the_image(tmp_path):
    """A dirty-tree check does not catch a tree that moved ON to further commits after the build —
    the likelier mistake, because builds take long enough that work continues while they run. Caught
    live on v23: one commit landed between build and publish, and the manifest would have named a
    recipe_commit that rebuilds into a different image (different rootfs -> RTMR1), so a miner
    following it would be rejected `unapproved_runtime` and conclude the owner was lying."""
    import subprocess

    mod = _publish_script()
    staged = tmp_path / "mkosi.extra/opt/koth/venv/lib/python3.12/site-packages/thirtyspokes/koth"
    staged.mkdir(parents=True)
    baked = staged / "epoch.py"
    at_head = subprocess.run(["git", "show", "HEAD:src/thirtyspokes/koth/epoch.py"],
                             capture_output=True).stdout

    baked.write_bytes(at_head)                       # image matches HEAD -> publishable
    mod._refuse_unless_head_built_this_image(str(tmp_path), dry_run=False)

    baked.write_bytes(at_head + b"\n# built from a commit that is no longer HEAD\n")
    with pytest.raises(SystemExit) as e:
        mod._refuse_unless_head_built_this_image(str(tmp_path), dry_run=False)
    assert "koth/epoch.py" in str(e.value) and "DIFFERENT image" in str(e.value)

    # no build dir -> cannot verify, so warn rather than block (the pre-existing behaviour)
    mod._refuse_unless_head_built_this_image(None, dry_run=False)


def test_cascade_stops_escalating_when_the_task_runs_out_of_time():
    """One task's worst case used to exceed the whole epoch: every rung is an HTTP call with its own
    timeout and retries, and a task entering low can climb several rungs. Measured live (epoch
    76734): five tasks in 131s, then ~950s on the sixth, and the epoch was lost — a proof that misses
    its epoch is unrecoverable, since its nonce is stale."""
    from thirtyspokes.koth import harness as H
    from thirtyspokes.koth.benchmarks import real_suite

    bench = next(b for b in real_suite() if b.name == "math")
    REJECTED = "I cannot solve this."
    assert not H.verifier_ok(REJECTED, bench), "precondition: this answer must make the ladder climb"

    pool = H.pool_models()
    order = H.rung_order(pool, H.price_of)
    clock = {"t": 0.0}

    def slow(model, messages, params):
        clock["t"] += 200.0                      # each rung overruns the whole per-task budget
        return REJECTED

    answer, used = H.run_cascade(0, "q", bench, pool, order, slow, {},
                                 budget_s=150.0, now=lambda: clock["t"])
    assert len(used) == 1, f"escalation must stop once the budget is blown, got {len(used)} rungs"
    assert answer == REJECTED, "a truncated cascade banks what it has — never an empty answer"

    # unbudgeted, the same task climbs the WHOLE ladder: that is the epoch-killing behaviour
    clock["t"] = 0.0
    _, unbounded = H.run_cascade(0, "q", bench, pool, order, slow, {},
                                 budget_s=float("inf"), now=lambda: clock["t"])
    assert len(unbounded) == len(order)

    # the FIRST call is never skipped, however far behind the clock already is
    clock["t"] = 10_000.0
    _, used2 = H.run_cascade(0, "q", bench, pool, order, slow, {},
                             budget_s=150.0, now=lambda: clock["t"])
    assert len(used2) == 1, "a task with no call has no answer — that is worse than a cheap one"


def test_preflight_bound_comes_from_the_harness_budget_not_a_guess():
    from thirtyspokes.koth import doctor
    from thirtyspokes.koth.harness import RUN_BUDGET_S

    status, _, detail = doctor.check_slice_fits_epoch(2)
    assert f"{RUN_BUDGET_S:.0f}s run budget" in detail and status == doctor.OK


def test_slice_agreement_admits_when_it_cannot_see_the_other_sources(tmp_path, monkeypatch):
    """A containerised validator has the package in site-packages, so the CLI defaults and the cron
    are not on disk. The check used to report a confident `ok` after comparing the process against
    itself — which reads as "all four components agree" when nothing was compared at all."""
    from thirtyspokes.koth import doctor

    monkeypatch.setenv("KOTH_REPO_ROOT", str(tmp_path))     # empty: no CLI/cron files to read
    monkeypatch.setenv("KOTH_UNIT_DIR", str(tmp_path))      # and no deployed units either
    status, _n, detail = doctor.check_slice_agreement(2)
    assert status == doctor.WARN and "against itself" in detail

    # a real disagreement still FAILS, seen or unseen sources notwithstanding
    assert doctor.check_slice_agreement(99)[0] == doctor.FAIL

    # and a DEPLOYED unit counts as a source: the check listed a cron script and kept passing after
    # the reference moved to systemd, validating a file nothing runs
    (tmp_path / "koth-reference.service").write_text("ExecStart=x --n-per-bench 2 --loop\n")
    ok, _n2, d2 = doctor.check_slice_agreement(2)
    assert ok == doctor.OK and "unit koth-reference=2" in d2


def test_a_single_hung_call_cannot_outlive_the_task_budget():
    """The escalation gate alone did NOT fix the epoch-killer, and epoch 76738 proved it: five tasks
    in 174s, then one call hung ~900s and the run missed its epoch WITH the budget in place.

    The gate checks the clock BEFORE issuing a call, so it cannot interrupt one already in flight —
    and a logical call was bounded only by the client timeout (120s) x its retries (4) x this
    backend's own retry loop (3), i.e. ~24 minutes, longer than an epoch. The bound has to reach the
    call, so the runtime arms `MeteringProxy.task_deadline` and the backend honours it.
    """
    from thirtyspokes.tee.runtime import ALLOWED_PARAMS, MeteringProxy

    seen = {}

    class SlowBackend:
        def complete(self, model, messages, params):
            seen.update(params)
            return "answer", 1, 1, 0.0

    proxy = MeteringProxy(SlowBackend())
    proxy.call_model("m", [{"role": "user", "content": "q"}], {"max_tokens": 8})
    assert "_timeout" not in seen, "no deadline armed -> no bound imposed"

    import time
    proxy.task_deadline = time.monotonic() + 30.0
    proxy.call_model("m", [{"role": "user", "content": "q"}], {"max_tokens": 8})
    assert 0 < seen["_timeout"] <= 30.0, f"call must inherit the task's remaining budget: {seen}"

    # a MINER must not be able to widen its own bound: the deadline is injected after the allow-list
    assert "_timeout" not in ALLOWED_PARAMS and "timeout" not in ALLOWED_PARAMS
    proxy.task_deadline = time.monotonic() + 5.0
    proxy.call_model("m", [{"role": "user", "content": "q"}], {"_timeout": 9999, "timeout": 9999})
    assert seen["_timeout"] <= 5.0, "agent-supplied timeout must never survive the filter"


def test_run_budget_fits_the_validators_grace_window_not_just_the_epoch():
    """A proof must beat the validator's GRACE (85 blocks ~1020s), which is tighter than the epoch.
    At 150s/task the arithmetic landed 14 seconds late on 76738 and went unscored."""
    from thirtyspokes.koth.harness import MIN_TASK_S, RUN_BUDGET_S, task_budget

    overhead = 40 + 30 + 20          # boot, attest/emit, upload
    assert RUN_BUDGET_S + overhead < 85 * 12, "worst-case run cannot beat the grace point"

    # cheap tasks hand their slack to the expensive one — the whole reason the budget is shared
    assert task_budget(780.0, 5) == 780.0 - 5 * MIN_TASK_S
    assert task_budget(606.0, 0) == 606.0, "the last task may use everything that is left"

    # ...but nobody can starve the tasks after it
    assert task_budget(50.0, 5) == MIN_TASK_S
    assert task_budget(-10.0, 3) == MIN_TASK_S, "an overrun still leaves later tasks able to answer"


def test_the_operator_stops_a_run_that_cannot_beat_the_published_grace():
    """A proof landing after the validators' grace point is complete, valid, attested — and unscored.
    The miner cannot tell: its own run prints "uploaded". Measured on 76738, missed by 14 seconds.

    So the owner publishes the grace point and the operator sizes each attempt against the time
    actually left, abandoning a run that provably cannot be scored instead of burning a VM."""
    from thirtyspokes.koth.epoch import EPOCH_BLOCKS
    from thirtyspokes.koth.gcp_operator import _attempt_deadline

    epoch, grace = 100, 85
    open_block = epoch * EPOCH_BLOCKS

    # plenty of room -> the operator's own setting still governs
    assert _attempt_deadline(900.0, grace, open_block, epoch) == 900.0
    # late start -> capped at what is really left, minus the upload margin
    assert _attempt_deadline(900.0, grace, open_block + 40, epoch) == 45 * 12.0 - 30.0
    # past the grace point -> a floor, not a negative or absurd deadline
    assert _attempt_deadline(900.0, grace, open_block + 90, epoch) == 60.0
    # nothing published -> unchanged behaviour
    assert _attempt_deadline(900.0, None, open_block + 40, epoch) == 900.0


def test_governance_check_rejects_a_grace_the_run_budget_cannot_beat(monkeypatch):
    """If the owner publishes a grace point the harness cannot beat, EVERY honest miner produces a
    valid proof that lands too late — the most expensive kind of misconfiguration, because nothing
    looks broken from either side."""
    from thirtyspokes.koth import doctor
    from thirtyspokes.koth.harness import pool_models
    from thirtyspokes.koth.runtime import runtime_measurement

    def fake_chain(record):
        class C:
            def __init__(self, *a, **k):
                pass

            def owner_measurements(self):
                return record
        return C

    base = {"runtime_measurements": [runtime_measurement()], "pool_allow_list": list(pool_models())}

    import thirtyspokes.subnet.chain as chain_mod
    monkeypatch.setattr(chain_mod, "BittensorChain", fake_chain({**base, "grace_blocks": 20}))
    status, _n, detail = doctor.check_governance(526, "test", "w", "hk")
    assert status == doctor.FAIL and "grace" in detail

    monkeypatch.setattr(chain_mod, "BittensorChain", fake_chain({**base, "grace_blocks": 85}))
    assert doctor.check_governance(526, "test", "w", "hk")[0] == doctor.OK

    monkeypatch.setattr(chain_mod, "BittensorChain", fake_chain(base))     # nothing published
    assert doctor.check_governance(526, "test", "w", "hk")[0] == doctor.OK


def test_reference_cells_are_bounded_and_retried():
    """One hung or failed cell used to cost the whole epoch's reference, not just its own row: a row
    needs EVERY model, so with a small slice a single bad cell drops every row and nothing is
    published. Validators then silently fall back to absolute accuracy — routing is not scored at
    all, and nothing says so. Observed live on 76745: 13/14 cells, `nothing to publish`."""
    import inspect

    from thirtyspokes.koth import reference

    src = inspect.getsource(reference.build)
    assert "_timeout" in src, "each cell must carry a wall-clock bound to the provider call"
    assert "cell_timeout_s" in src and "deadline_s * 0.6" in src, (
        "the bound must leave room for a legitimately slow cell (~360s measured), or every slow row "
        "fails every epoch — a permanent failure traded for a rare one")
    assert "_attempts" in src, "a bounded cell is worth one retry inside the deadline"


def test_validator_announces_when_it_is_not_scoring_routing():
    """No pool reference means the scalar degrades to absolute accuracy — the subnet keeps paying and
    the log keeps looking healthy while the thing it exists to measure goes unmeasured. That used to
    be visible only as two ABSENT fields, which nobody notices."""
    import inspect

    from thirtyspokes.koth import neuron

    src = inspect.getsource(neuron)
    assert "reference=MISSING" in src, "a silent fallback must announce itself"
    assert "NOT routing" in src


def test_build_freshness_catches_a_container_running_stale_code(tmp_path, monkeypatch):
    """`docker compose restart` reuses the existing image, so a fix can look deployed and not be.
    It happened twice in one day — once caught only because the runtime measurement had changed, once
    caught by noticing a log line was missing. When the change does not move the measurement, nothing
    notices at all."""
    from thirtyspokes.koth import doctor

    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text("a" * 40 + "\n")
    monkeypatch.setenv("KOTH_REPO_ROOT", str(tmp_path))

    monkeypatch.setenv("KOTH_BUILD_COMMIT", "a" * 40)
    assert doctor.check_build_freshness()[0] == doctor.OK

    monkeypatch.setenv("KOTH_BUILD_COMMIT", "b" * 40)
    status, _n, detail = doctor.check_build_freshness()
    assert status == doctor.WARN and "does NOT rebuild" in detail

    monkeypatch.setenv("KOTH_BUILD_COMMIT", "unknown")
    assert doctor.check_build_freshness()[0] == doctor.WARN

    monkeypatch.delenv("KOTH_BUILD_COMMIT")          # not containerised
    assert doctor.check_build_freshness()[0] == doctor.OK


def test_a_bounded_call_is_bounded_in_total_not_per_attempt():
    """The bug that survived the first fix and killed epoch 76746 on the supposedly-bounded engine.

    `timeout` bounds ONE SDK attempt, not the call: with the client's max_retries=3 a single
    create() can still take 4 x timeout, so the call looked bounded and was not. Five tasks finished
    in 128s and the sixth never returned. Asserting that a timeout is *passed* — which the earlier
    test did — cannot catch this; the assertion has to be on total elapsed time.
    """
    import time

    from thirtyspokes.gateway.gateway import OpenRouterBackend

    calls = {"n": 0, "opts": None}

    class FakeCompletions:
        def create(self, **kw):
            calls["n"] += 1
            time.sleep(float(kw["timeout"]))     # every attempt burns its whole timeout
            raise RuntimeError("provider hung")

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

        def with_options(self, **kw):
            calls["opts"] = kw
            return self

    be = OpenRouterBackend.__new__(OpenRouterBackend)
    be._client = FakeClient()
    be._price_fn = lambda m: (0.0, 0.0)

    t0 = time.monotonic()
    text, tin, tout, cost = be.complete("m", [{"role": "user", "content": "q"}],
                                        {"max_tokens": 8, "_timeout": 1.0})
    elapsed = time.monotonic() - t0

    assert calls["opts"] == {"max_retries": 0}, "the SDK's own retries must not multiply the bound"
    assert elapsed < 3.0, f"a 1s budget took {elapsed:.1f}s — the call is not actually bounded"
    assert text == "" and cost == 0.0, "an unanswerable rung degrades; it does not crash the epoch"


def test_a_run_of_hanging_calls_still_finishes_and_still_emits_a_proof(monkeypatch):
    """End-to-end proof of the bound, against the failure that killed four live epochs.

    The unit tests cover one call; this drives the real `run_router` with a pool where EVERY rung
    hangs, and asserts the run both stays bounded and still produces a complete proof. Scaled to
    seconds so it is a genuine wall-clock test rather than a mock of one.

    Written only after shipping two images that each 'fixed' this and did not: reasoning about the
    mechanism is what produced those, and a timing assertion is what would have stopped them.
    """
    import time

    from thirtyspokes.koth import harness as H
    from thirtyspokes.koth.benchmarks import real_suite
    from thirtyspokes.koth.miner import routing_artifact
    from thirtyspokes.koth.runtime import KOTHRuntime, mock_vendor_platform

    monkeypatch.setattr(H, "RUN_BUDGET_S", 6.0)
    monkeypatch.setattr(H, "MIN_TASK_S", 1.0)
    HANG = 20.0

    class HangingPool:
        """IGNORES the granted bound, which is the whole point.

        The first version of this test slept only what it was granted — proving the runtime *asks*
        for a bound while assuming the compliance that epoch 76751 disproved: an httpx read timeout
        does not fire on a response that trickles, so the call neither returned nor timed out. A
        bound that depends on the thing being bounded is not a bound.
        """
        calls = 0

        def complete(self, model, messages, params):
            HangingPool.calls += 1
            assert params.get("_timeout") is not None, "the runtime must still grant a bound"
            time.sleep(HANG)              # ...and ignore it completely
            return "", 0, 0, 0.0

    suite = real_suite()
    # A real head, so the entry rungs are the ones a live miner would pick. Which rung it picks does
    # not matter here — every rung hangs — but a synthetic head would make the test prove less.
    weights = pathlib.Path("/root/koth-miner-work/weights.npz")
    if not weights.exists():
        pytest.skip("needs a trained head; the unit tests cover the bound without one")
    art = routing_artifact(weights.read_bytes())

    t0 = time.monotonic()
    proof, _trace = KOTHRuntime(HangingPool(), mock_vendor_platform()).run_router(
        art.weights, hotkey="5Test", epoch=1, nonce="n", suite=suite,
        n_per_bench=1, pool=H.pool_models(), price_of=H.price_of)
    elapsed = time.monotonic() - t0

    assert len(proof.results) == len(suite), "a bounded run must still answer every task"
    # budget + the watchdog's per-task slack; fixed cost, negligible at production scale (6 tasks
    # against a 780s budget) but dominant here, which is why it is stated rather than fudged.
    ceiling = H.RUN_BUDGET_S + len(suite) * 5.0 + 15.0
    unbounded = len(suite) * HANG
    assert elapsed < ceiling, f"UNBOUNDED: {elapsed:.1f}s against a {ceiling:.0f}s ceiling"
    assert elapsed < unbounded, f"{elapsed:.1f}s is not under the {unbounded:.0f}s of an unbounded run"


def test_reference_skips_an_epoch_it_cannot_reach_in_time():
    """A reference arriving after the grace point describes a slice already scored without it: the
    calls are spent and the epoch falls back to absolute accuracy anyway. Observed on 76755."""
    from thirtyspokes.koth.epoch import EPOCH_BLOCKS
    from thirtyspokes.koth.reference import seconds_until_scored

    epoch, grace, deadline = 100, 85, 600.0
    open_block = epoch * EPOCH_BLOCKS

    # early in the epoch: the whole deadline fits before scoring
    assert seconds_until_scored(epoch, grace, open_block + 5) == 80 * 12.0
    assert seconds_until_scored(epoch, grace, open_block + 5) > deadline

    # 80 blocks in: 60s left, so a 600s run would land long after the epoch was scored
    assert seconds_until_scored(epoch, grace, open_block + 80) == 5 * 12.0
    assert seconds_until_scored(epoch, grace, open_block + 80) < deadline

    # past the grace point entirely -> negative, still less than any deadline
    assert seconds_until_scored(epoch, grace, open_block + 95) < 0

    # no published grace -> no opinion, the caller proceeds as before
    assert seconds_until_scored(epoch, None, open_block + 80) is None


def test_a_missing_reference_is_not_cached_forever():
    """The owner publishes DURING the epoch, so a lookup before that moment must not poison it.

    Measured on 76756: the reference was published at 22:25, the epoch was scored at 22:27, and the
    validator still reported `reference=MISSING` — it had cached the miss from an earlier poll and
    never looked again."""
    from thirtyspokes.koth import reference
    from thirtyspokes.koth.runtime import SUITE_VERSION

    published = {}
    calls = {"n": 0}

    def fake_fetch(epoch, nonce, **kw):
        calls["n"] += 1
        if epoch not in published:
            raise FileNotFoundError("not published yet")
        return published[epoch]

    class Chain:
        def owner_hotkey(self): return "5Owner"

    monkey = reference.fetch
    reference.fetch = fake_fetch
    try:
        read = reference.chain_reader(Chain(), n_per_bench=2, verify_sig=lambda *a, **k: True)
        assert read(50, "n") is None, "not published yet"
        published[50] = {"epoch": 50, "nonce": "n", "n_per_bench": 2,
                         "suite_version": SUITE_VERSION, "scores": [[1.0]], "costs": [[0.1]]}
        assert read(50, "n") is not None, "a later publish must be picked up, not cached away"
        assert calls["n"] == 2, "the miss must not have been served from cache"
        assert read(50, "n") is not None and calls["n"] == 2, "a HIT is still cached"
    finally:
        reference.fetch = monkey


def test_the_operator_reaps_vms_a_crash_left_behind():
    """A SIGKILLed operator cannot run its `finally`, so its CVM keeps billing. The per-attempt
    cleanup only removes the SAME name, so once the epoch advances the orphan is never touched.
    A fault-injection restart found two of them still running seven hours later."""
    import inspect

    from thirtyspokes.koth import gcp_operator

    src = inspect.getsource(gcp_operator.reap_orphans)
    assert "instances" in src and "list" in src, "must enumerate, not guess names"
    # scoped to THIS miner's own hash prefix, so it can never delete another operator's VM
    assert "^koth-{prefix}-" in src or "name~^koth-" in src
    # and it must be called at startup, where it can see orphans from earlier epochs
    assert "reap_orphans(args.zone, prefix)" in inspect.getsource(gcp_operator.main)


def test_every_created_vm_carries_a_cloud_enforced_lifetime():
    """Miner-side cleanup has a hole no miner-side code can close: `finally` does not run on SIGKILL,
    and the startup reaper only helps if the operator comes back. A dead host or an uninstalled miner
    leaves the CVM billing forever — two ran for seven hours after the operator that made them died.

    The cloud bound is the only layer that survives the miner disappearing entirely."""
    import inspect

    from thirtyspokes.koth import gcp_operator

    src = inspect.getsource(gcp_operator._boot_once)
    assert "--max-run-duration" in src and "instance-termination-action=DELETE" in src
    # ...and it must not be able to cut a legitimate run short: the operator's own attempt deadline
    # is 900s, so the cloud bound has to sit comfortably above it
    assert gcp_operator.MAX_VM_MINUTES * 60 > 900 * 1.5


def test_the_reference_loop_bounds_its_own_child():
    """The loop exists to stop a run outliving its epoch. If the child can wedge the LOOP, every
    later epoch silently loses its reference — and systemd sees a live process, so nothing restarts.
    Reintroducing the bug one level up is the easiest mistake to make here."""
    import inspect

    from thirtyspokes.koth import reference

    src = inspect.getsource(reference._loop)
    assert "timeout=deadline" in src, "the child run must be bounded"
    assert "TimeoutExpired" in src and "moving to the next epoch" in src, (
        "a wedged child must cost one epoch, not every epoch after it")


def test_a_validator_scoring_earlier_than_the_published_grace_is_refused(monkeypatch):
    """Miners size their runs against the grace the owner publishes. A validator scoring EARLIER
    misses proofs that met that contract exactly and disqualifies honest miners as `no_proof` — the
    miner did everything right and the log blames the miner."""
    from thirtyspokes.koth import doctor
    from thirtyspokes.koth.harness import pool_models
    from thirtyspokes.koth.runtime import runtime_measurement

    rec = {"runtime_measurements": [runtime_measurement()], "pool_allow_list": list(pool_models()),
           "grace_blocks": 85}

    class C:
        def __init__(self, *a, **k): pass
        def owner_measurements(self): return rec

    import thirtyspokes.subnet.chain as chain_mod
    monkeypatch.setattr(chain_mod, "BittensorChain", C)

    assert doctor.check_governance(526, "test", "w", "hk", grace_blocks=85)[0] == doctor.OK
    status, _n, detail = doctor.check_governance(526, "test", "w", "hk", grace_blocks=50)
    assert status == doctor.FAIL and "no_proof" in detail
    # later than published is only a warning: it waits longer, it does not miss anything
    assert doctor.check_governance(526, "test", "w", "hk", grace_blocks=95)[0] == doctor.WARN
    # and with no grace passed (e.g. the standalone CLI) the check is unchanged
    assert doctor.check_governance(526, "test", "w", "hk")[0] == doctor.OK


def test_the_reign_is_order_independent_so_two_validators_cannot_disagree():
    """Two validators only pay the same miner if they rank identically from the same facts.

    They receive candidates in whatever order their own chain reads and downloads produced, so any
    reliance on input order — a sort keyed on score alone, a set iteration — would make the winner
    depend on which validator scored. That divergence would be silent: each validator's own log looks
    correct, and miners are simply paid differently depending on who counted.

    Observed agreeing live on epochs 76765-76767; this pins it as an invariant rather than luck.
    """
    import itertools

    from thirtyspokes.reign import Reign, Submission

    # deliberately includes an exact score TIE, which is where input order would leak through
    subs = [
        Submission(miner_id="a", hotkey="hkA", commit_block=100, score=0.50),
        Submission(miner_id="b", hotkey="hkB", commit_block=90, score=0.50),
        Submission(miner_id="c", hotkey="hkC", commit_block=120, score=0.75),
        Submission(miner_id="d", hotkey="hkD", commit_block=80, score=0.10),
    ]
    live = {s.miner_id for s in subs}

    results = []
    for perm in itertools.permutations(subs):
        r = Reign()
        out = r.update(list(perm), deregistered=set(), live=live)
        results.append((tuple(out.slots), tuple(sorted(out.weights.items())), out.coronation))

    assert len(set(results)) == 1, (
        f"the reign depends on candidate ORDER: {len(set(results))} distinct outcomes across "
        f"permutations of the same facts — two validators would pay different miners")


def test_an_abandoned_task_is_a_wrong_answer_not_fraud():
    """The watchdog's whole purpose is to turn a stalled provider into a lost TASK instead of a lost
    EPOCH. That failed on 76768: the runtime abandoned task 6 after 738s and uploaded a proof, and
    the validator rejected the entire proof as `no_pool_call` because that task had no metered call.
    The miner did everything right and still earned nothing.

    Answering without calling is fraud. Failing without calling is not.
    """
    import inspect

    from thirtyspokes.koth import validator as V

    src = inspect.getsource(V)
    assert "answered.get(tid)" in src, "the rule must key on whether an ANSWER was produced"

    # the guard still bites when an answer appears with no call behind it
    calls = {"t1": 1, "t2": 0}
    answered = {"t1": "42", "t2": "1729"}          # t2 answered with no call -> fraud
    assert any(calls.get(t, 0) < 1 and answered.get(t) for t in ("t1", "t2"))

    answered_empty = {"t1": "42", "t2": ""}        # t2 abandoned, no answer -> honest
    assert not any(calls.get(t, 0) < 1 and answered_empty.get(t) for t in ("t1", "t2"))


def test_a_question_containing_its_own_answer_is_not_laundering():
    """Epoch 76799 disqualified ALL FOUR miners — three of mine and one independent — as `laundered`
    in the same epoch, burned the emissions, and the log accused every one of them of cheating.

    gsm8k-798's answer is 20 and its question mentions 20. `extract_number` takes the last number,
    found it in the prompt, and concluded the agent had fed the pool an answer it already held. On
    the ROUTING path that inference is impossible: the harness sends the owner's task text verbatim
    and the miner supplies only a rung index, so it never authors a prompt at all.
    """
    from thirtyspokes.koth.verify import _grounded_one

    calls = [{"prompt": "Billy earns $20 more than Sally. How much...?", "response": "The answer is 20"}]

    # free-agent path: the agent DID write that prompt, so prompt-provenance still applies
    assert _grounded_one("20", calls, "number", agent_authored_prompts=True) == (False, "laundered")

    # routing path: the prompt is the owner's question, so this is just a question containing its
    # own answer — and the token does appear in the response, which is what grounding means
    assert _grounded_one("20", calls, "number", agent_authored_prompts=False) == (True, "ok")

    # answering without the pool is still caught on BOTH paths — the fix narrows one rule, not all
    no_pool = [{"prompt": "Billy earns $20 more than Sally.", "response": "I am not sure."}]
    assert _grounded_one("20", no_pool, "number", agent_authored_prompts=False) == (False, "ungrounded")


def test_the_attempt_deadline_cannot_truncate_a_run_that_was_about_to_succeed():
    """Epoch 76806: the watchdog abandoned task 6 at its budget, the run COMPLETED at 856s kernel
    time, and the operator killed it at its 900s deadline before collecting the proof.

    The watchdog made this visible rather than causing it — before, such runs died as hangs, so a
    full-budget run was never seen finishing. The deadline has to clear the harness worst case with
    room for GCP provisioning (which precedes the kernel clock, so it is invisible in serial
    timestamps) and the operator's poll granularity."""
    from thirtyspokes.koth.benchmarks import real_suite
    from thirtyspokes.koth.gcp_operator import _attempt_deadline
    from thirtyspokes.koth.harness import RUN_BUDGET_S

    import inspect
    src = inspect.getsource(_attempt_deadline)
    assert "grace" in src, "the published grace must govern when available"

    tasks = 2 * len(real_suite())
    worst = RUN_BUDGET_S + tasks * 5 + 40 + 30 + 15      # budget, slack, boot, emit, poll
    import thirtyspokes.koth.gcp_operator as G
    default = [a for a in inspect.getsource(G.main).split("\n") if "--attempt-deadline" in a][0]
    value = float(default.split("default=")[1].split(",")[0])
    assert value > worst, f"default {value}s does not clear the {worst:.0f}s harness worst case"

    # and the grace-derived cap must still be the binding constraint early in an epoch
    from thirtyspokes.koth.epoch import EPOCH_BLOCKS
    room = _attempt_deadline(value, 85, 100 * EPOCH_BLOCKS, 100)
    assert room < value, "the published grace should bind before the static ceiling"


def test_build_freshness_reads_packed_refs(tmp_path, monkeypatch):
    """A repack — `git gc`, or a history rewrite — moves refs out of refs/heads/ into packed-refs.
    Reading only the loose path then fails, so the check went blind exactly when history changed,
    which is when it matters most. Observed after purging a file from this repo's history."""
    from thirtyspokes.koth import doctor

    sha = "c" * 40
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "packed-refs").write_text(f"# pack-refs with: peeled fully-peeled sorted\n{sha} refs/heads/main\n")
    monkeypatch.setenv("KOTH_REPO_ROOT", str(tmp_path))

    monkeypatch.setenv("KOTH_BUILD_COMMIT", sha)
    assert doctor.check_build_freshness()[0] == doctor.OK, "packed refs must resolve"

    monkeypatch.setenv("KOTH_BUILD_COMMIT", "d" * 40)
    assert doctor.check_build_freshness()[0] == doctor.WARN, "a stale image must still be caught"

    # neither loose nor packed -> say so rather than claim a pass
    (git / "packed-refs").unlink()
    assert doctor.check_build_freshness()[0] == doctor.WARN


def test_an_abandoned_task_does_not_fail_grounding_either():
    """The second half of the same bug. Narrowing `no_pool_call` (76768) left `ungrounded` catching
    the identical proof by another route on 76816: a watchdog-abandoned task has an empty answer, its
    token is None, no response matches None, and it reads as 'answered without the pool'.

    An empty answer is not an answer. It scores zero like any wrong one; it is not fraud."""
    from thirtyspokes.koth.verify import _grounded_one

    # abandoned: no answer, no calls -> honest failure on both paths
    assert _grounded_one("", [], "code", agent_authored_prompts=False) == (True, "ok")
    assert _grounded_one("   ", [], "number", agent_authored_prompts=True) == (True, "ok")

    # a REAL answer with no pool call behind it is still ungrounded — the rule keeps its teeth
    calls = [{"prompt": "q", "response": "unrelated"}]
    assert _grounded_one("42", calls, "number", agent_authored_prompts=False) == (False, "ungrounded")


def test_a_stalled_reference_builder_is_a_blocking_fault(monkeypatch):
    """The outage that changes what the subnet pays for without stopping anything.

    `_load_reference` degrades to the legacy `Q_lcb - cost` scalar on purpose, so when the owner's
    builder stops the validator keeps scoring and keeps setting weights — on a different quantity.
    The epoch line does say `reference=MISSING`, but mainnet still ran 48 epochs that way: a field in
    a running process's log is not a gate. This makes it one.

    Checking the BUCKET rather than the process is the point: a wedged builder is still `active`.
    """
    from thirtyspokes.koth import doctor

    class _Sub:
        def __init__(self, block): self._b = block
        def get_current_block(self): return self._b

    class _Chain:
        block = 8749100
        def __init__(self, *a, **k): self.subtensor = _Sub(_Chain.block)

    published = {"epochs": [87442, 87443]}

    class _Api:
        def __init__(self, *a, **k): pass
        def list_bucket_tree(self, *a, **k):
            return [type("E", (), {"path": f"reference/{e}-abc.json"})()
                    for e in published["epochs"]]

    import thirtyspokes.subnet.chain as chainmod
    import huggingface_hub
    monkeypatch.setattr(chainmod, "BittensorChain", _Chain)
    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)

    # chain at epoch 87491, newest record 87443 -> 48 epochs behind: the real mainnet outage
    status, name, msg = doctor.check_reference_freshness(99, "finney", "w", "h")
    assert status == doctor.FAIL, "a 48-epoch-stale reference must block, not warn"
    assert "87443" in msg and "legacy" in msg, f"must name the record and the consequence: {msg}"

    # current epoch still being built is normal, not a fault
    published["epochs"] = [87490, 87491]
    assert doctor.check_reference_freshness(99, "finney", "w", "h")[0] == doctor.OK

    # never published at all is the same failure, and must not read as "fine, nothing to compare"
    published["epochs"] = []
    assert doctor.check_reference_freshness(99, "finney", "w", "h")[0] == doctor.FAIL


def test_the_preflight_always_terminates_even_when_the_chain_does_not(monkeypatch):
    """A preflight that hangs is worse than the fault it looks for — it blocks the daemon.

    Measured live: the same `BittensorChain(...)` construction took 3s on a good endpoint and >9
    minutes on a bad one, because the SDK's own retry ladder runs before any timeout we configure is
    reachable. Both chain-reading checks must give up and say so.
    """
    import time

    from thirtyspokes.koth import doctor

    monkeypatch.setattr(doctor, "CHAIN_READ_TIMEOUT_S", 0.3)

    def _never_answers():
        time.sleep(30)
        return "unreachable"

    t0 = time.time()
    val, err = doctor._bounded(_never_answers, timeout=0.3)
    assert val is None and isinstance(err, TimeoutError)
    assert time.time() - t0 < 5, "the bound did not hold"

    # an exception inside the thread is reported, not raised on a thread nobody is joining
    val, err = doctor._bounded(lambda: 1 / 0)
    assert val is None and isinstance(err, ZeroDivisionError)

    # and a healthy read still passes its value through
    assert doctor._bounded(lambda: 42) == (42, None)


def test_an_unknown_network_name_blocks_instead_of_quietly_disabling_every_chain_check():
    """`NETWORK=mainnet` in a live `.env` — and mainnet is called `finney`.

    Worse than a hard failure: an unknown name fails DNS inside `Subtensor(...)`, both chain checks
    catch it and downgrade to "cannot check", warnings never block, and the preflight prints "no
    blocking problems" having verified neither governance nor the reference.
    """
    from thirtyspokes.koth import doctor

    status, _n, detail = doctor.check_network_name("mainnet")
    assert status == doctor.FAIL
    assert "finney" in detail, "must say what the right name is, not just that this one is wrong"

    assert doctor.check_network_name("finney")[0] == doctor.OK
    assert doctor.check_network_name("test")[0] == doctor.OK
    assert doctor.check_network_name("wss://my.endpoint")[0] == doctor.FAIL
