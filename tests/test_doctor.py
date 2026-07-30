"""Startup preflight: the deployment-shape checks that broke live runs (koth/doctor.py)."""

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
    from thirtyspokes.koth.harness import TASK_BUDGET_S

    status, _, detail = doctor.check_slice_fits_epoch(2)
    assert f"{TASK_BUDGET_S:.0f}s budget" in detail
    assert doctor.check_slice_fits_epoch(16)[0] == doctor.FAIL


def test_slice_agreement_admits_when_it_cannot_see_the_other_sources(tmp_path, monkeypatch):
    """A containerised validator has the package in site-packages, so the CLI defaults and the cron
    are not on disk. The check used to report a confident `ok` after comparing the process against
    itself — which reads as "all four components agree" when nothing was compared at all."""
    from thirtyspokes.koth import doctor

    monkeypatch.setenv("KOTH_REPO_ROOT", str(tmp_path))     # empty: no CLI/cron files to read
    status, _n, detail = doctor.check_slice_agreement(2)
    assert status == doctor.WARN and "against itself" in detail

    # a real disagreement still FAILS, seen or unseen sources notwithstanding
    assert doctor.check_slice_agreement(99)[0] == doctor.FAIL
