"""Which docker daemon grades, and proving it before a window opens (§8b.9, the sandbox-host seam).

WHAT THESE TESTS PROTECT. §8b.9 splits the hosts by what they hold: the Orchestrator holds the chain
hotkey, the R2 credentials and the gateway key, and the sandbox fleet — which runs the graders, i.e.
executes programs a miner's model wrote — holds no credentials at all. `V3_DOCKER_HOST` is the only
thing in the code that can express that split, so the properties below are security properties and
not configuration conveniences:

  * **All FIVE docker call sites follow it, not just `run`.** `image_present` and its batched twin
    `images_present` are the expensive ones to get wrong: answered by the local daemon they report a
    corpus that is provisioned somewhere the grading never happens, and `--pull never` converts that
    into a refusal at task 1 of a window.
    `_kill` is here for the same reason with the opposite cost — a killer aimed at the wrong daemon
    leaves the hung container running on the right one, which is the leak `_kill` exists to close.
  * **The measured bind-mount bug is made LOUD.** A second daemon promotes testnet 526's silent
    "the bind mounts nothing" from an exception to the default case, so `preflight` refuses instead
    of returning a plausible number — and it checks the mount by CONTENT, because identical path
    strings are exactly what that failure looked like from this side.
  * **The configuration is DECLARED, never defaulted into.** Measured on the owner's box: with the
    seam unset, grading ran on the root daemon as real host root and `preflight` returned exit 0
    with a success line. So an unset seam now executes nothing, `local` is how a single-box
    developer says the same thing out loud, and the three tests below pin which entry points became
    strict (`run`, `available`, `preflight`) and which did not (importing this module, `workspace`,
    `docker_host` — i.e. every offline path, which is why `simulate` and the rest of the suite are
    untouched).
  * **A cap that vanished and a harness failure that scored are both refusals.** A daemon without a
    delegated cgroup accepts `--memory`/`--pids-limit`/`--cpus` and ignores them; docker's own
    start failures arrive on 126/127 as well as 125, where a real program's exit code also lives.

NO DAEMON IS NEEDED TO RUN ANY OF THIS, deliberately rather than for convenience: the seam exists for
deployments with a SECOND daemon and no test box has one, so every assertion here is either about the
command line this module builds or about a fake daemon returning the answers a real misconfiguration
returns. The live path stays covered by `test_real_benchmarks.py` behind `needs_docker`.
"""

from __future__ import annotations

import pathlib
import subprocess
import time

import pytest

from thirtyspokes.v3.benchmarks import sandbox

HOST = "unix:///run/user/1000/docker.sock"      # a rootless daemon beside the Orchestrator
AMBIENT = "unix:///var/run/docker.sock"         # the root daemon this seam exists to stay off


@pytest.fixture(autouse=True)
def _a_client_but_no_daemon(monkeypatch):
    """A docker CLI on PATH and nothing else real. Every test below fakes the daemon itself, so this
    only removes the one difference between a box with docker installed and a box without."""
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.delenv(sandbox.DOCKER_HOST_ENV, raising=False)
    monkeypatch.delenv("DOCKER_HOST", raising=False)


class _FakeDaemon:
    """A docker client whose daemon behaves as constructed, spliced in at the `subprocess` boundary.

    BELOW `sandbox.run` rather than instead of it: the code under test builds and issues its real
    command line — `--pull never`, `--network none`, the `-v` this whole seam is about — and the
    assertions read it back. A fake that replaced `run` would agree with any implementation.
    """

    def __init__(self, *, reachable: bool = True, has_image: bool = True, mount: str = "resolves",
                 caps: str = "true true true"):
        self.reachable, self.has_image, self.mount, self.caps = reachable, has_image, mount, caps
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append(argv)
        sub = argv[3] if argv[1] == "--host" else argv[1]
        if sub == "version":
            return subprocess.CompletedProcess(
                argv, 0 if self.reachable else 1, stdout="27.1.1\n" if self.reachable else "",
                stderr="" if self.reachable else "Cannot connect to the Docker daemon")
        if sub == "info":
            # `docker info --format '{{.MemoryLimit}} {{.PidsLimit}} {{.CPUCfsQuota}}'` — the
            # daemon's own answer about the cgroup controllers it holds.
            return subprocess.CompletedProcess(argv, 0, stdout=self.caps + "\n", stderr="")
        if sub == "image":
            # `docker image inspect` prints one object per argument it RESOLVED and skips the rest,
            # so a daemon that has them echoes the names back as tags and one that has none prints
            # nothing — while exiting non-zero either way, which is the thing the census must not
            # read. `--format` and its value are not names.
            names = argv[argv.index("inspect") + 1:]
            if names[:1] == ["--format"]:
                names = names[2:]
            return subprocess.CompletedProcess(
                argv, 0 if self.has_image else 1,
                stdout="".join(f"{name}\n" for name in names) if self.has_image else "")
        if sub == "run":
            source = pathlib.Path(argv[argv.index("-v") + 1].split(":")[0])
            probe = source / "probe.txt"
            if self.mount == "resolves":
                # What a shared filesystem does: the container sees the bytes we just wrote. Runs
                # that are not the preflight put no probe there and get an empty log, as they do.
                kwargs["stdout"].write(probe.read_bytes() if probe.exists() else b"")
            elif self.mount == "nothing":       # testnet 526: the bind resolves to an empty dir
                kwargs["stdout"].write(b"cat: /w/probe.txt: No such file or directory\n")
                return subprocess.CompletedProcess(argv, 1)
            else:                               # a path that exists there and holds something else
                kwargs["stdout"].write(b"0123456789abcdef\n")
            return subprocess.CompletedProcess(argv, 0)
        return subprocess.CompletedProcess(argv, 0)

    def subcommands(self) -> set[str]:
        return {argv[3] if argv[1] == "--host" else argv[1] for argv in self.calls}


def _install(monkeypatch, daemon: _FakeDaemon) -> _FakeDaemon:
    monkeypatch.setattr(sandbox.subprocess, "run", daemon)
    return daemon


# --- which daemon each call site talks to ---------------------------------------------------------
def test_the_local_declaration_aims_nothing_anywhere_new(monkeypatch, tmp_path):
    """`local` must be byte-for-byte the command line this module issued before the seam existed. A
    single-box deployment is a real one — a developer, a dev-net box — and the declaration is there
    to make it SAY so, not to make it impossible. What it costs is one word; what it buys is that
    silence stops meaning "the root daemon, probably"."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, sandbox.DOCKER_HOST_LOCAL)
    daemon = _install(monkeypatch, _FakeDaemon())
    sandbox.available()
    sandbox.image_present("v3-probe:1")
    sandbox.run("v3-probe:1", ["true"], mount=tmp_path, log_path=tmp_path / "log", timeout=5)

    assert all("--host" not in argv for argv in daemon.calls)
    assert all(argv[0] == sandbox.DOCKER for argv in daemon.calls)
    # ...and `available` costs nothing at all here: a client on PATH IS the local-socket answer.
    assert "version" not in daemon.subcommands()


def test_the_seam_aims_run_image_present_available_and_the_killer_at_one_daemon(monkeypatch,
                                                                                tmp_path):
    """All five, because the module's promise is that nothing it executes runs on the Orchestrator's
    own daemon — one call site left behind is one place a miner's program still runs on the box that
    holds the hotkey, or a container that leaks on the box that does."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    monkeypatch.setattr(sandbox, "KILL_WINDOW_SECONDS", 0.0)      # one pass, no sleep
    daemon = _install(monkeypatch, _FakeDaemon())

    sandbox.available()
    sandbox.image_present("v3-probe:1")
    sandbox.images_present(["v3-probe:1", "v3-probe:2"])
    sandbox.run("v3-probe:1", ["true"], mount=tmp_path, log_path=tmp_path / "log", timeout=5)
    sandbox._kill("v3-deadbeef")

    assert daemon.subcommands() == {"version", "image", "run", "rm"}
    # `--host` is a GLOBAL flag: after `docker` and before the subcommand, or the client rejects it.
    for argv in daemon.calls:
        assert argv[:3] == [sandbox.DOCKER, "--host", HOST], argv


def test_image_present_answers_about_the_grading_daemon_and_not_the_local_one(monkeypatch):
    """The named hazard, stated as the difference it makes. The corpus is provisioned on the fleet;
    asking the Orchestrator's daemon reports the opposite answer, and `--pull never` then spends the
    window discovering it one task at a time."""
    def only_the_fleet_has_it(argv, **kwargs):
        argv = list(argv)
        on_fleet = argv[:3] == [sandbox.DOCKER, "--host", HOST]
        sub = argv[3] if argv[1] == "--host" else argv[1]
        return subprocess.CompletedProcess(argv, 0 if (on_fleet or sub == "version") else 1)

    monkeypatch.setattr(sandbox.subprocess, "run", only_the_fleet_has_it)
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, sandbox.DOCKER_HOST_LOCAL)
    assert sandbox.image_present("v3-corpus:1") is False       # local: the local daemon answers
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    assert sandbox.image_present("v3-corpus:1") is True


def test_the_corpus_census_costs_one_round_trip_per_chunk_and_not_two_per_image(monkeypatch):
    """THE LATENT BILL. `image_present` calls `available()`, and `available()` issues a live `docker
    version` whenever the seam names a daemon of its own — so a per-image census pays TWO round
    trips per task, and R2E-Gym's 6812 tasks are ~13,600 of them at the 0.84 s a multiplexed ssh
    `docker` was measured to cost (§1.3a): 2000-4000 s of a window's clock on housekeeping. Nothing
    pays it today only because `usable_ids` has no caller in `src/` yet, which is the moment to fix
    it rather than after it is wired into window building.

    The `docker version` probe is NOT what was removed — a client on PATH says nothing about a
    daemon on another host, which is its whole job. What was removed is paying for it per image, so
    the assertion is on the SHAPE of the traffic: one probe, then one inspect per chunk."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    monkeypatch.setattr(sandbox, "IMAGE_CENSUS_CHUNK", 4)
    daemon = _install(monkeypatch, _FakeDaemon())
    names = [f"v3-corpus:{n}" for n in range(10)]

    assert sandbox.images_present(names) == set(names)

    assert [argv[3] for argv in daemon.calls] == ["version", "image", "image", "image"]
    chunks = [argv[argv.index("--format") + 2:] for argv in daemon.calls if argv[3] == "image"]
    assert [len(chunk) for chunk in chunks] == [4, 4, 2]
    assert sorted(name for chunk in chunks for name in chunk) == sorted(names)


def test_the_census_reads_the_returned_objects_and_never_the_exit_code(monkeypatch):
    """`docker image inspect a b c` exits NON-ZERO if any one of them is missing, and the partly
    provisioned corpus is the only case that ever happens on the real box — 23 of 731 on SWE-bench
    Pro, ~32 of 6812 on R2E-Gym. A census that read the code would answer "none present" for the
    whole chunk and a window would be built from nothing.

    Nor can the objects be matched back positionally: docker prints one per argument it RESOLVED, in
    order, and simply skips the rest — which is why presence is decided by the tags the daemon
    reports for the images that did come back."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)

    class _HasSome(_FakeDaemon):
        def __call__(self, argv, **kwargs):
            argv = list(argv)
            if argv[3] == "image":
                self.calls.append(argv)
                return subprocess.CompletedProcess(argv, 1, stdout="v3-corpus:b\n")
            return super().__call__(argv, **kwargs)

    _install(monkeypatch, _HasSome())
    assert sandbox.images_present(["v3-corpus:a", "v3-corpus:b"]) == {"v3-corpus:b"}


def test_the_census_asks_no_daemon_at_all_when_none_is_declared_or_nothing_is_wanted(monkeypatch):
    """The same rule as `image_present`, kept at the batched entry point: undeclared, the honest
    answer about the grading daemon's corpus is "nothing", without a probe (§8b.9). The empty
    request is here because a benchmark whose census is skipped must not still pay the `docker
    version` the seam requires."""
    daemon = _install(monkeypatch, _FakeDaemon())
    assert sandbox.images_present(["v3-corpus:a"]) == set()
    assert daemon.calls == []

    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    assert sandbox.images_present([]) == set()
    assert daemon.calls == []


def test_a_v3_host_outranks_an_ambient_docker_host(monkeypatch, tmp_path):
    """The reason for a second name. `DOCKER_HOST` is ambient — rootless docker's own setup prints
    it, a `docker context` sets it, a shell profile exports it — so the v3 setting has to be the
    more specific statement and win. `--host` is the top of the client's precedence ladder."""
    monkeypatch.setenv("DOCKER_HOST", AMBIENT)
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    daemon = _install(monkeypatch, _FakeDaemon())

    sandbox.run("v3-probe:1", ["true"], mount=tmp_path, log_path=tmp_path / "log", timeout=5)
    assert daemon.calls[-1][:3] == [sandbox.DOCKER, "--host", HOST]
    assert AMBIENT not in daemon.calls[-1]


def test_available_refuses_a_configured_daemon_that_does_not_answer(monkeypatch):
    """A client on PATH is evidence about a LOCAL socket and about nothing else. With the seam
    configured, `available` has to reach the daemon it named, or a corpus check passes on a machine
    that cannot grade a single task."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    _install(monkeypatch, _FakeDaemon(reachable=False))
    assert sandbox.available() is False


# --- the declaration: silence is not consent ------------------------------------------------------
def test_run_executes_nothing_at_all_until_a_grading_host_is_declared(monkeypatch, tmp_path):
    """THE MEASURED FAILURE, closed. With the seam unset on the owner's box, `run` graded on the ROOT
    daemon as real host root (`uid_map: 0 0 4294967295`) — the daemon that also runs several
    unrelated projects' production containers, on the machine holding the chain hotkey. So the check
    is BEFORE the command line is built: no subprocess at all, and the message names both ways to
    say which daemon means it."""
    daemon = _install(monkeypatch, _FakeDaemon())
    with pytest.raises(sandbox.SandboxError) as exc:
        sandbox.run("v3-probe:1", ["true"], mount=tmp_path, log_path=tmp_path / "log", timeout=5)

    assert daemon.calls == []                                    # nothing was executed anywhere
    message = str(exc.value)
    assert sandbox.DOCKER_HOST_ENV in message and sandbox.DOCKER_HOST_LOCAL in message
    assert "unix:///run/user/1000/docker.sock" in message        # the fleet's shape, not just a rule


def test_preflight_refuses_before_it_probes_anything_when_nothing_is_declared(monkeypatch):
    """The gate that green-lit the dangerous configuration: undeclared, this printed a success line
    and exited 0. The refusal is unconditional rather than conditional on a second daemon being
    reachable — reachability is a fact about the world at 3am, and the declaration is one about
    intent."""
    daemon = _install(monkeypatch, _FakeDaemon())
    with pytest.raises(sandbox.SandboxError, match=sandbox.DOCKER_HOST_ENV):
        sandbox.preflight("v3-probe:1")
    assert daemon.calls == []


def test_available_is_false_until_a_host_is_declared_and_true_once_it_is(monkeypatch):
    """`available` is what everything downstream asks before opening the grading path — the corpus
    census, and the live test suite's `needs_docker` skip. Undeclared it must answer no, without a
    probe: a client on PATH is exactly what made an unconfigured box look configured."""
    daemon = _install(monkeypatch, _FakeDaemon())
    assert sandbox.available() is False
    assert daemon.calls == []

    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, sandbox.DOCKER_HOST_LOCAL)
    assert sandbox.available() is True


def test_the_offline_paths_are_untouched_by_the_declaration(monkeypatch, tmp_path):
    """WHO IS DELIBERATELY NOT REFUSED. Importing this module, allocating a workspace and asking
    which host is configured execute nothing, so `simulate`, the mock benchmarks and every test that
    is not `needs_docker` keep working on a box that has never heard of docker. Grading is the thing
    that must be declared, not importing the module."""
    monkeypatch.setenv(sandbox.GRADE_DIR_ENV, str(tmp_path))
    daemon = _install(monkeypatch, _FakeDaemon())

    assert sandbox.declared() is False
    assert sandbox.docker_host() is None
    with sandbox.workspace() as (work, logs):
        assert work.exists() and work.parent == tmp_path
        assert logs.exists()
    assert daemon.calls == []


def test_local_is_a_declaration_and_not_a_host(monkeypatch, tmp_path):
    """`local` must never reach the client as `--host local`, which would be a dial to a daemon of
    that name. It is this module's word for "resolve it the way you always would", so an ambient
    `DOCKER_HOST` still stands underneath it — the difference from unset is the statement, not the
    endpoint."""
    monkeypatch.setenv("DOCKER_HOST", AMBIENT)
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, sandbox.DOCKER_HOST_LOCAL)
    daemon = _install(monkeypatch, _FakeDaemon())

    assert sandbox.docker_host() is None and sandbox.declared() is True
    sandbox.run("v3-probe:1", ["true"], mount=tmp_path, log_path=tmp_path / "log", timeout=5)
    assert "--host" not in daemon.calls[-1] and sandbox.DOCKER_HOST_LOCAL not in daemon.calls[-1]


# --- the launch gate: which machine, answered before a window opens -------------------------------
def test_the_launch_gate_refuses_an_undeclared_process_before_probing_anything(monkeypatch):
    """A daemon that starts undeclared does not fail at launch — it fails at every task, as an
    exclusion from both arms, after the window has paid for every worker call in it. So the gate
    is the same declaration `run` makes, moved to where it costs one line instead of a window."""
    daemon = _install(monkeypatch, _FakeDaemon())
    with pytest.raises(sandbox.SandboxError, match=sandbox.DOCKER_HOST_ENV):
        sandbox.check_grading_host()
    assert daemon.calls == []


def test_the_launch_gate_refuses_a_remote_daemon_with_no_shared_payload_directory(monkeypatch):
    """The bind-path problem, refused rather than discovered. `docker run -v` is resolved by the
    DAEMON, so with the endpoint on another machine and `V3_GRADE_DIR` unset the payload is a
    tempdir that exists only here: the bind mounts an empty directory and every task ends with no
    verdict. That is the one arrangement this process can rule out by reading its own environment,
    so it is ruled out BEFORE any probe — reachability is not what is wrong with it."""
    monkeypatch.delenv(sandbox.GRADE_DIR_ENV, raising=False)
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, "ssh://root@gpu-host")
    daemon = _install(monkeypatch, _FakeDaemon())

    with pytest.raises(sandbox.SandboxError) as exc:
        sandbox.check_grading_host()

    assert daemon.calls == []
    message = str(exc.value)
    assert sandbox.GRADE_DIR_ENV in message and "IDENTICAL" in message
    # ...and it must not claim to have checked the arrangement that CAN work: that one is by
    # content, it needs a container, and it is `preflight`'s.
    assert "does not check the one that can" in message


def test_a_remote_daemon_with_a_shared_payload_directory_is_admitted_and_named(monkeypatch,
                                                                                tmp_path):
    """The deployed arrangement (§8b.9): the process keeps the credentials here, its containers land
    over there, and the payload directory resolves at the same absolute path on both. The gate
    admits it and RETURNS the sentence an operator's log needs — which endpoint, which directory —
    because "grading is configured" and "grading is configured to run on that box" are different
    claims and only the second is what the split is about."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, "ssh://root@gpu-host")
    monkeypatch.setenv(sandbox.GRADE_DIR_ENV, str(tmp_path))
    daemon = _install(monkeypatch, _FakeDaemon())

    line = sandbox.check_grading_host()

    assert "ssh://root@gpu-host" in line and str(tmp_path) in line
    assert daemon.subcommands() == {"version"}, "one probe, no container"


def test_a_second_daemon_on_this_box_is_not_refused_for_lacking_a_grade_dir(monkeypatch):
    """`unix://` is deliberately absent from `REMOTE_ENDPOINT_SCHEMES`. A rootless daemon beside the
    Orchestrator shares the filesystem, so the path resolves; what may still fail is the UID reading
    a 0700 tempdir, and no environment read can answer that. Refusing it here would be this gate
    guessing at a question `preflight` measures by content."""
    monkeypatch.delenv(sandbox.GRADE_DIR_ENV, raising=False)
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    _install(monkeypatch, _FakeDaemon())

    assert HOST in sandbox.check_grading_host()


def test_the_launch_gate_refuses_a_grade_dir_that_is_not_a_directory_here(monkeypatch, tmp_path):
    """The other half of the payload seam, and it was unchecked. `V3_GRADE_DIR` naming a path that
    does not exist on THIS box passed the gate, and then every `workspace()` raised a bare
    `FileNotFoundError` from `tempfile` — not even a `SandboxError` — which `_GraderGuard` catches as
    broadly as any other, so task after task left both arms and the denominator (§8b.3) while the
    window spent its whole allowance on worker calls whose grades were all discarded. A typo in one
    flag and the window resolves nothing.

    It belongs here by this gate's own rule: which paths exist on this filesystem is a fact about
    this process's own environment, which is exactly the class of question the gate answers and
    `preflight`'s by-content probe is not needed for. Refused BEFORE the daemon probe, because a
    reachable daemon does not make an absent directory work.

    A FILE is refused too, not only an absent path: `tempfile.TemporaryDirectory(dir=…)` needs a
    directory, and `os.path.exists` would have passed the one case that reads as configured."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, sandbox.DOCKER_HOST_LOCAL)
    daemon = _install(monkeypatch, _FakeDaemon())

    for bad in (tmp_path / "not-created", _a_file(tmp_path)):
        monkeypatch.setenv(sandbox.GRADE_DIR_ENV, str(bad))
        with pytest.raises(sandbox.SandboxError, match="is not a directory on this box"):
            sandbox.check_grading_host()
        with pytest.raises(OSError):
            with sandbox.workspace():
                pass                        # the failure the refusal above is standing in front of
    assert daemon.calls == [], "refused before any probe: reachability is not the problem with it"

    monkeypatch.setenv(sandbox.GRADE_DIR_ENV, str(tmp_path))
    assert str(tmp_path) in sandbox.check_grading_host(), "a real directory is still admitted"


def test_the_launch_gate_refuses_a_log_dir_that_is_not_a_directory_here(monkeypatch, tmp_path):
    """The same window-eating failure as the grade directory's, through the other half of the same
    call: `workspace()` allocates under BOTH roots, so a typo in the newer variable raises a bare
    `OSError` inside every grade, each task leaves both arms and the denominator (§8b.3), and the
    window resolves nothing having spent everything. It is answerable from this process's own
    environment — this box's own filesystem — so it belongs at launch by this gate's own rule."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, sandbox.DOCKER_HOST_LOCAL)
    daemon = _install(monkeypatch, _FakeDaemon())

    for bad in (tmp_path / "not-created", _a_file(tmp_path)):
        monkeypatch.setenv(sandbox.LOG_DIR_ENV, str(bad))
        with pytest.raises(sandbox.SandboxError, match="is not a directory on this box"):
            sandbox.check_grading_host()
        with pytest.raises(OSError):
            with sandbox.workspace():
                pass                        # the failure the refusal above is standing in front of
    assert daemon.calls == [], "refused before any probe: reachability is not the problem with it"

    monkeypatch.setenv(sandbox.LOG_DIR_ENV, str(tmp_path))
    assert sandbox.check_grading_host(), "a real directory is still admitted"


def _a_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "a-file"
    path.write_text("")
    return path


def test_the_launch_gate_refuses_a_declared_daemon_that_does_not_answer(monkeypatch, tmp_path):
    """Declared, coherent, and down. Every task would raise and leave both arms and the denominator,
    which is §8b.3 working correctly and producing a window that resolved nothing — so the daemon
    refuses to start rather than spending the money to find out."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, "ssh://root@gpu-host")
    monkeypatch.setenv(sandbox.GRADE_DIR_ENV, str(tmp_path))
    _install(monkeypatch, _FakeDaemon(reachable=False))

    with pytest.raises(sandbox.SandboxError, match="no docker daemon answered"):
        sandbox.check_grading_host()


def test_the_single_box_declaration_passes_the_launch_gate_and_says_so(monkeypatch):
    """`local` is a real deployment and the gate must not make it impossible — it makes it AUDIBLE.
    The returned line is what a reader of the validator's log sees, and on this configuration it
    says that this box is the one executing miner-written code."""
    monkeypatch.delenv(sandbox.GRADE_DIR_ENV, raising=False)
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, sandbox.DOCKER_HOST_LOCAL)
    daemon = _install(monkeypatch, _FakeDaemon())

    line = sandbox.check_grading_host()

    assert sandbox.DOCKER_HOST_LOCAL in line and f"{sandbox.GRADE_DIR_ENV} unset" in line
    assert daemon.calls == [], "a client on PATH IS the local answer; nothing to probe"


# --- a harness failure on a code a program can also exit with (§8b.3) -----------------------------
class _ExitingRun(_FakeDaemon):
    """A daemon whose `run` exits with a given code after writing a given log."""

    def __init__(self, code: int, log: bytes):
        super().__init__()
        self.code, self.log = code, log

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        if (argv[3] if argv[1] == "--host" else argv[1]) == "run":
            self.calls.append(argv)
            kwargs["stdout"].write(self.log)
            return subprocess.CompletedProcess(argv, self.code)
        return super().__call__(argv, **kwargs)


# The measured one (exit 126) and the exec-not-found one (127), both docker's own voice.
MOUNT_DENIED = (b"docker: Error response from daemon: error while creating mount source path "
                b"'/tmp/tmpe_j8hi46/payload': mkdir /tmp/tmpe_j8hi46/payload: permission denied\n")
NO_ENTRYPOINT = (b'docker: Error response from daemon: failed to create task for container: '
                 b'exec: "python": executable file not found in $PATH: unknown\n')


@pytest.mark.parametrize("code, log", [(126, MOUNT_DENIED), (127, NO_ENTRYPOINT)])
def test_a_start_failure_docker_reports_as_126_or_127_raises_instead_of_scoring(monkeypatch,
                                                                                tmp_path, code, log):
    """MEASURED: a rootless daemon that cannot create the bind source exits 126, which slipped past
    the 125 guard and came back as a `SandboxResult` — i.e. the validator's own breakage scored as
    the miner's answer, which §8b.3/§6.3c forbid. The exit code alone cannot say so (a program may
    exit 126), so the discriminator is the pair: one of those codes AND a line docker itself
    wrote."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    _install(monkeypatch, _ExitingRun(code, log))
    with pytest.raises(sandbox.SandboxError) as exc:
        sandbox.run("v3-probe:1", ["true"], mount=tmp_path, log_path=tmp_path / "log", timeout=5)
    assert "could not start" in str(exc.value)


@pytest.mark.parametrize("code, log", [
    (126, b"/w/build.sh: line 3: ./configure: Permission denied\n"),
    (127, b"make: cc: No such file or directory\n"),
    (126, b"the suite printed docker: Error response from daemon: mid-line and carried on\n")])
def test_a_program_that_really_exited_126_is_still_a_score(monkeypatch, tmp_path, code, log):
    """The other half, and the reason the constant was not simply widened to 126/127: those are
    ordinary exits for a program that RAN, and a task wrongly excluded is as bad as one wrongly
    scored — it deletes evidence from both arms. Third case: the marker is only docker's when docker
    wrote the LINE, so a suite quoting the phrase mid-line stays a result."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    _install(monkeypatch, _ExitingRun(code, log))
    result = sandbox.run("v3-probe:1", ["true"], mount=tmp_path, log_path=tmp_path / "log",
                         timeout=5)
    assert result.exit_code == code


# --- the killer, on the link the timeout is about -------------------------------------------------
class _WedgedRemote(_FakeDaemon):
    """A daemon that answers `run` with a timeout and then never answers `rm` at all.

    The pair matters: `_kill` is only ever reached from inside `run`'s `except TimeoutExpired`, and
    the reason the container blew its clock is usually the reason the removal will not answer either.
    `subprocess.run`'s real semantics are modelled — with no `timeout=` the client waits forever, and
    with one it waits that long and raises — because a stub that returned early would pass against
    the unbounded code this test exists to refuse.
    """

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        sub = argv[3] if argv[1] == "--host" else argv[1]
        if sub in ("run", "rm"):
            self.calls.append(argv)
            limit = kwargs.get("timeout")
            if limit is None:
                pytest.fail(f"`docker {sub}` was issued against {HOST} with no timeout")
            time.sleep(min(limit, 0.05))
            raise subprocess.TimeoutExpired(argv, limit)
        return super().__call__(argv, **kwargs)


def test_the_killer_is_bounded_when_the_remote_daemon_never_answers(monkeypatch, tmp_path):
    """MEASURED, and it is the transport's failure rather than the container's: `_docker` prepends
    `--host`, so every removal `_kill` issues is a network operation over the same link that just
    failed to finish a container. Unbounded, one wedged ssh link blocked `_kill` for good — INSIDE
    `run`'s except, so the §8b.3 exclusion was never raised and the arm stopped forever, where the
    per-duel clock cannot reach it because that is checked between tasks."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    monkeypatch.setattr(sandbox, "KILL_WINDOW_SECONDS", 0.05)
    daemon = _install(monkeypatch, _WedgedRemote())

    with pytest.raises(sandbox.SandboxError) as exc:
        sandbox.run("v3-probe:1", ["true"], mount=tmp_path, log_path=tmp_path / "log", timeout=0.05)

    assert "excluded from both arms" in str(exc.value)
    assert any((argv[3] if argv[1] == "--host" else argv[1]) == "rm" for argv in daemon.calls)


def test_a_killer_that_fails_does_not_replace_the_timeout_that_brought_it_in(monkeypatch, tmp_path):
    """`_kill`'s docstring promises best effort; it has to be written down rather than meant. A
    client that is simply gone raised out of the killer and REPLACED the `SandboxError` with its own
    exception — the same task, bucketed by whichever handler happened to catch a `FileNotFoundError`
    instead of by §8b.3."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    monkeypatch.setattr(sandbox, "KILL_WINDOW_SECONDS", 0.0)

    class _Vanished(_FakeDaemon):
        def __call__(self, argv, **kwargs):
            argv = list(argv)
            if (argv[3] if argv[1] == "--host" else argv[1]) == "rm":
                raise FileNotFoundError(2, "No such file or directory: 'docker'")
            raise subprocess.TimeoutExpired(argv, kwargs.get("timeout") or 1.0)

    _install(monkeypatch, _Vanished())
    with pytest.raises(sandbox.SandboxError):
        sandbox.run("v3-probe:1", ["true"], mount=tmp_path, log_path=tmp_path / "log", timeout=0.05)


# --- the caps, the codes and the capture: what a mutation battery found untested -------------------
def test_the_grading_command_line_carries_every_cap_the_box_is_made_of(monkeypatch, tmp_path):
    """MUTATION-FOUND GAP: deleting `--security-opt no-new-privileges`, or the `--memory` pair, from
    `run` left the ENTIRE suite green (1420 tests).

    `--network none`, `--pull never` and the `:ro` bind were each pinned by a test and each died
    under mutation; the other four were pinned nowhere, so the sandbox was four flags away from being
    a `docker run` with a volume and nothing else — silently, and on the one path that executes a
    program a miner's model wrote. Asserted as ONE list rather than four assertions because the
    property is the SET: a box missing any one of them is not the box this module documents, and a
    per-flag test is a per-flag omission waiting to happen. Read off `run`'s own argv rather than
    `preflight`'s, because `run` is what grades."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    daemon = _install(monkeypatch, _FakeDaemon())

    sandbox.run("v3-probe:1", ["true"], mount=tmp_path, log_path=tmp_path / "log", timeout=5)

    ran = next(argv for argv in daemon.calls if "run" in argv)
    assert ran[ran.index("--pull") + 1] == "never"
    assert ran[ran.index("--network") + 1] == "none"
    assert ran[ran.index("--memory") + 1] == sandbox.MEMORY
    # Equal to `--memory`, or the cap degrades to RAM-only and the workload swaps straight past it.
    assert ran[ran.index("--memory-swap") + 1] == sandbox.MEMORY
    assert ran[ran.index("--pids-limit") + 1] == str(sandbox.PIDS_LIMIT)
    assert ran[ran.index("--cpus") + 1] == sandbox.CPUS
    assert ran[ran.index("--security-opt") + 1] == "no-new-privileges"
    assert ran[ran.index("-v") + 1] == f"{tmp_path}:{sandbox.MOUNT}:ro"
    assert "--rm" in ran


def test_the_plain_125_start_failure_raises_rather_than_scoring(monkeypatch, tmp_path):
    """MUTATION-FOUND GAP: `result.exit_code == DOCKER_RUN_FAILED` could be deleted outright and the
    suite stayed green — the AMBIGUOUS half (126/127) was tested and the unambiguous half was not.

    125 is the one code that is never a workload's, so this is the simplest case of §8b.3 and the one
    a missing image under `--pull never` arrives on: the operator's omission, which must take the task
    out of both arms rather than score the miner zero for it."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    _install(monkeypatch, _ExitingRun(
        sandbox.DOCKER_RUN_FAILED,
        b"docker: Error response from daemon: No such image: v3-probe:1\n"))
    with pytest.raises(sandbox.SandboxError) as exc:
        sandbox.run("v3-probe:1", ["true"], mount=tmp_path, log_path=tmp_path / "log", timeout=5)
    assert "could not start" in str(exc.value)


def test_the_captured_output_is_the_end_of_the_log_and_not_the_beginning(tmp_path):
    """MUTATION-FOUND GAP, and the gap is that the cover is BEHIND DOCKER: `_tail` could seek to 0
    instead of to the last `MAX_CAPTURE_BYTES` and the whole offline suite stayed green.

    The property is not untested — `test_real_benchmarks.py` catches it — but that test is
    `@needs_docker`, i.e. one of the 22 skipped on any box without a daemon, which is every box this
    seam is developed on. WHICH END it is is the property `_tail`'s docstring states: a driver prints
    its verdict when it FINISHES, so a capture taken from the front names the wrong thing in every
    exception this module raises, and `tools._payload` hunts the observation fence in exactly this
    string."""
    log = tmp_path / "log"
    log.write_bytes(b"FIRST" + b"." * (sandbox.MAX_CAPTURE_BYTES * 2) + b"LAST")

    captured = sandbox._tail(log)

    assert captured.endswith("LAST")
    assert "FIRST" not in captured
    assert len(captured.encode()) <= sandbox.MAX_CAPTURE_BYTES


# --- the preflight: every way this can be misconfigured, named ------------------------------------
def test_preflight_refuses_a_daemon_that_does_not_answer(monkeypatch):
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    _install(monkeypatch, _FakeDaemon(reachable=False))
    with pytest.raises(sandbox.SandboxError) as exc:
        sandbox.preflight("v3-probe:1")
    assert sandbox.DOCKER_HOST_ENV in str(exc.value) and HOST in str(exc.value)


@pytest.mark.parametrize("caps, flags", [
    ("false false false", ("--memory", "--pids-limit", "--cpus")),
    ("true false true", ("--pids-limit",))])
def test_preflight_refuses_a_daemon_that_would_ignore_the_resource_caps(monkeypatch, caps, flags):
    """MEASURED: a rootless daemon started without a delegated cgroup — a bare `dockerd-rootless.sh`,
    no systemd user session — ACCEPTS `--memory`, `--pids-limit` and `--cpus` and applies none of
    them, saying so only in a startup warning. Those flags are the only thing stopping one graded
    task from taking the validator down mid-window, so a sandbox whose caps silently vanished is not
    a sandbox and the fix is named where the failure is reported."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    _install(monkeypatch, _FakeDaemon(caps=caps))
    with pytest.raises(sandbox.SandboxError) as exc:
        sandbox.preflight("v3-probe:1")
    message = str(exc.value)
    assert all(flag in message for flag in flags)
    assert "loginctl enable-linger" in message


def test_preflight_refuses_when_cap_support_cannot_be_read_at_all(monkeypatch):
    """Fails CLOSED. Treating "I could not tell" as "the caps are fine" is the exact shape of the bug
    above: the daemon answers, the flags are accepted, and nothing enforces them. The cost of being
    wrong this way is one loud preflight against an unexpected client; the other way it is a window
    of uncapped containers on the machine that sets weights."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    _install(monkeypatch, _FakeDaemon(caps="template: :1: unexpected field"))
    with pytest.raises(sandbox.SandboxError, match="cap support"):
        sandbox.preflight("v3-probe:1")


def test_preflight_refuses_an_image_the_grading_daemon_does_not_have(monkeypatch):
    """`--pull never` is deliberate (a missing image must not spend an arm's wall clock), which makes
    provisioning an operator obligation — and the obligation is against the FLEET's daemon."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    _install(monkeypatch, _FakeDaemon(has_image=False))
    with pytest.raises(sandbox.SandboxError) as exc:
        sandbox.preflight("v3-corpus:1")
    message = str(exc.value)
    assert "--pull never" in message and "pull v3-corpus:1" in message and HOST in message


@pytest.mark.parametrize("mount", ["nothing", "somebody-elses-bytes"])
def test_preflight_catches_the_bind_that_does_not_resolve(monkeypatch, mount):
    """THE MEASURED BUG (testnet 526), made loud instead of silent. A daemon that resolves `-v` in
    another namespace mounts nothing and the driver is simply absent; a second daemon makes that the
    DEFAULT rather than the exception. The stale-bytes case is why the probe is checked by content:
    a path can exist on the far side and be a different directory, which no comparison of path
    strings can see."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    _install(monkeypatch, _FakeDaemon(mount=mount))
    with pytest.raises(sandbox.SandboxError) as exc:
        sandbox.preflight("v3-probe:1")
    message = str(exc.value)
    assert sandbox.GRADE_DIR_ENV in message      # the fix, named where the failure is reported
    assert sandbox.MOUNT in message


def test_preflight_proves_the_mount_through_the_real_grading_command_line(monkeypatch):
    """What passing has to mean: the same command grading issues, against the daemon grading uses,
    round-tripping bytes that could not have come from anywhere else. Asserted on the argv rather
    than on the return value, because a preflight that proved a SIMPLER command would pass on a
    deployment where `--network none` or the read-only bind is what breaks the mount."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    daemon = _install(monkeypatch, _FakeDaemon())

    line = sandbox.preflight("v3-probe:1")

    ran = next(argv for argv in daemon.calls if "run" in argv)
    assert ran[:3] == [sandbox.DOCKER, "--host", HOST]
    assert ran[ran.index("--pull") + 1] == "never"
    assert ran[ran.index("--network") + 1] == "none"
    assert ran[ran.index("-v") + 1].endswith(f":{sandbox.MOUNT}:ro")
    assert "27.1.1" in line and HOST in line and "v3-probe:1" in line


def test_preflight_mints_a_fresh_token_every_time(monkeypatch):
    """A constant probe would be satisfied by a stale file left in a shared grade directory by the
    last run — which is the one thing that survives on the far side of a bind that used to work."""
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, HOST)
    seen = []

    class _Recording(_FakeDaemon):
        def __call__(self, argv, **kwargs):
            argv = list(argv)
            if (argv[3] if argv[1] == "--host" else argv[1]) == "run":
                source = pathlib.Path(argv[argv.index("-v") + 1].split(":")[0])
                seen.append((source / "probe.txt").read_text())
            return super().__call__(argv, **kwargs)

    _install(monkeypatch, _Recording())
    sandbox.preflight("v3-probe:1")
    sandbox.preflight("v3-probe:1")
    assert len(seen) == 2 and seen[0] != seen[1]
