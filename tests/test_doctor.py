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
