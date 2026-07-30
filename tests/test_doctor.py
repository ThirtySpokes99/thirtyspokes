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
