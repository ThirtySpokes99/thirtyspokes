"""The Docker seam — every real grader runs untrusted worker output in a throwaway box (M2a exit 5).

WHY A GRADER IS THE MOST DANGEROUS CODE PATH IN THE SUBNET. Grading LiveCodeBench is *executing* a
program a model wrote, and grading SWE-bench is applying a model's patch to a repository and running
its test suite — on the single owner-run machine that holds the subnet's entire scoring authority
(D1). That is the same threat model that makes `.bin` uploads refused (§1.1.2), arriving through a
different door: the artifact is refused for containing code, and then the *answers* it routes for
are code and we run them anyway, because there is no other way to grade them. So the box is the
mechanism, not an ops detail.

`--network none` is the load-bearing flag and it buys two separate things. A graded program must not
phone home, and — the reason it is not merely hygiene — a program that could reach the network could
fetch the expected outputs and score 1.0 without solving anything. Measured precedent for the rest of
the caps: nothing here bounds what a submitted program *tries* to do, so `--memory`, `--pids-limit`
and `--cpus` are what stop one task from taking the validator down mid-window, and `--pull never` is
what stops a missing image from spending an arm's two-hour wall clock (§8b.2) on the owner's own
housekeeping. Images are provisioned before a window opens; that is an operator obligation, and
`image_present` is here so it can be checked rather than discovered at task 173.

A SANDBOX FAILURE RAISES. IT NEVER BECOMES A ZERO (§8b.3, §6.3c). Docker dying is a fact about the
validator, so such a task is excluded from BOTH arms and from the denominator — which the caller can
only do if the failure reaches it. Everything else this module learns is returned as data, and the
adapter decides, because only the adapter knows its own verdict marker: a non-zero exit code is
ordinary (`git apply` refusing a malformed patch is a fact about the miner's answer), while an
absent verdict line is not.

TIMEOUT IS A SANDBOX FAILURE HERE, AND THAT IS A DELIBERATE CHANGE FROM `koth/lcb.py`, WHICH SCORED
IT 0.0. The two cases a container-level timeout mixes are "the submitted program hung" (a fact about
the miner's answer) and "the validator's clock was too tight for this test suite" (a fact about the
owner), and from outside the box they are the same event. §8b.3 resolves the tie in one direction
only — scoring it zero injects the owner's infrastructure noise into a miner's result, and an
exclusion is symmetric, counted and published. The repair for the case that IS the miner's is to
bound the workload *inside* the container, where the two are still distinguishable: LCB's driver
gives each test case its own timeout and still prints a verdict, so a hanging program comes back as
an honest `0 of 12` and never reaches this clock. A benchmark that cannot do that (SWE-bench's test
suites have no per-test bound) pays for it in dropped tasks instead of in corrupted scores.

THE BIND-MOUNT PATH MUST EXIST IN BOTH NAMESPACES, and this is the one bug in here that was measured
rather than reasoned about. `docker run -v` is resolved by the DAEMON, so when the validator is
itself containerised and talks to the host daemon through a mounted socket, a path inside this
container does not exist on the host: the bind silently mounts nothing, the driver is absent, and
grading fails. Measured on testnet 526 — the container had a working docker client (`docker version`
reported the host server) and could still not grade a single code task (`koth/lcb.py`). `workspace`
therefore allocates under `V3_GRADE_DIR`, which the operator points at a directory bind-mounted at
the IDENTICAL path on both sides.

THE LOG IS NOT PAYLOAD, AND UNDER A SHARED GRADE DIRECTORY THAT DISTINCTION IS HALF THE COST OF A
READ. Only `mount` is bind-mounted; the container's log is a host-side `stdout=` redirect that no
daemon ever opens, so it has no business on the storage the other machine sees. Measured
(the sandbox record §1.3a, corrected 2026-09-02): one tool read issues ~30 filesystem round trips —
mkdtemp, payload mkdir, two writes, then the log's create/write/stat/tail/full re-read, then rmtree
— and about half of them are the log's. On the deployed sshfs mount that harness overhead is
**1652 ms per read against 0.24 ms local** at 94.66 ms per round trip, ~1650 s an arm at ~1000 reads,
of which the log is ~800 s. So the log gets its own root, `V3_LOG_DIR`, defaulting to `tempfile`'s
own — CONFIGURABLE rather than hardcoded `/tmp`, because an operator who set `V3_GRADE_DIR` did it
to get off a small filesystem and a test suite's log runs to megabytes. It is also strictly a
containment improvement: less of what grading writes is visible on the filesystem the far side can
read.

WHICH DAEMON GRADES IS A SEAM, BECAUSE §8b.9 SPLITS THE HOSTS BY WHAT THEY HOLD. The Orchestrator
builds windows, decides verdicts and sets weights, so it holds the chain hotkey, the R2 credentials
and the gateway key; the sandbox fleet runs the agentic tool calls and the graders and holds *no
credentials at all*. This module is the reason that split is not an ops preference: what runs here is
a program a miner's model wrote, and on one box a container escape lands on the machine with sole
authority over emissions. `V3_DOCKER_HOST` is what makes the split expressible — every docker
invocation below (`run`, `image_present`, `images_present`, `available`, `_kill`) carries it as an
explicit `--host`, so the process that holds the keys can grade without executing anything on its
own daemon.

WHY A SECOND NAME RATHER THAN PLAIN `DOCKER_HOST`. The client already honours `DOCKER_HOST`, and
`V3_DOCKER_HOST=local` says "resolve it the way you always would" — no flag is added and an ambient
setting stands. What the second name buys is that the split becomes a STATEMENT instead of an
inheritance. `DOCKER_HOST` is set by rootless docker's own setup instructions, by a `docker context`,
by a shell profile, and it is inherited by every child process — so an unset one reads identically as
"deliberately local" and "nobody has configured this yet", and §8b.9 is precisely an assertion about
which machine executed the miner's code. It is passed as `--host` because that is the top of the
client's precedence ladder (`--host` > `DOCKER_HOST` > `DOCKER_CONTEXT` > the current context): the
v3-specific setting is the more specific statement and must not lose to an ambient one.

AN UNSET SEAM IS NOT A CONFIGURATION, IT IS A MISSING ONE — SO NOTHING EXECUTES UNTIL ONE IS SPOKEN.
Measured on the owner's own box: with `V3_DOCKER_HOST` unset, grading ran on the ROOT daemon as real
host root (`/proc/self/uid_map` = `0 0 4294967295`, no user namespace) — the daemon that also runs
several unrelated projects' production containers, on the machine that holds the chain hotkey — and
`preflight` returned exit 0 with a success line. A seam whose *default* value is the catastrophic one
is a seam that documents the split rather than enforcing it, so `run` and `preflight` now refuse
until the variable says something. `local` is the whole escape hatch a single-box developer needs,
and it costs one word said OUT LOUD instead of by omission. The refusal is unconditional rather than
conditional on a second daemon being reachable: reachability is a fact about the world at 3am, the
declaration is a statement of intent, and a check that passes when the fleet is briefly down is not a
check. WHO THIS REFUSES, EXACTLY: `run` (the only function here that executes anything), `available`
(which answers "may this process grade", and is what the live test suite skips on) and `preflight`.
Importing this module, `workspace()`, `docker_host()` and every offline path — `simulate`, the mock
benchmarks, the whole test suite bar the `needs_docker` ones — touch none of those and are unchanged.

ALL THREE ENTRY POINTS FOLLOW IT, NOT JUST `run`. `image_present` against the wrong daemon is the
expensive one — it answers about images the grading daemon does not have, and `--pull never` then
turns a corpus that checked out clean into a refusal at task 1. `available` grows a reachability
probe when the seam is configured, because a client on PATH implies a local socket and implies
nothing whatever about a daemon on another host or another uid.

POINTING AT A SECOND DAEMON PROMOTES THE BIND-MOUNT BUG ABOVE FROM AN EXCEPTION TO THE DEFAULT: two
daemons share a path only if they share a filesystem, and a rootless daemon runs as a uid that
cannot read the 0700 directory `workspace()` just created. That is why `preflight` exists and why it
RAISES — a seam that cannot be honoured must refuse rather than hand back a plausible number, which
is §8b.3's rule applied at configuration time, where it costs one line of output instead of a window.

WHICH DAEMON IS NOT THE SAME QUESTION AS WHICH PROCESS, AND THE SEAM ONLY ANSWERS THE FIRST. Moving
`V3_DOCKER_HOST` to another machine moves the *daemon*; the payload stays here, because it is a
tempdir on this filesystem. So a remote endpoint is only a working arrangement when `V3_GRADE_DIR`
names storage both machines see at the IDENTICAL absolute path — and with it unset, that arrangement
is not merely unproven, it is KNOWN not to work. `check_grading_host` refuses exactly that case, at
launch, before a window opens: it is the one configuration this module can rule out by reading its
own environment. Everything else about the mount is a by-content question and stays `preflight`'s,
which is why the refusal says so rather than implying the rest has been checked.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import secrets
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

DOCKER = "docker"

# Where `workspace()` is mounted inside the container. Read-only: a grader driver reads its payload
# from here and writes nothing, so the one directory the host and the container share cannot be used
# by the graded program to reach anything the host keeps beside it — including this run's own log,
# which lives outside the mount for that reason.
MOUNT = "/w"

# The operator's escape hatch for the measured bind-mount bug above. Unset is the ordinary case
# (`tempfile` picks its own root); set, it must name a directory mounted at the same path on both
# sides of the daemon.
GRADE_DIR_ENV = "V3_GRADE_DIR"

# Where the container's LOG goes, which is deliberately a different question. The log is never bind
# mounted, so it must not be allocated with the payload: under a shared `V3_GRADE_DIR` that cost
# about half of a read's filesystem round trips (module docstring). Unset means `tempfile`'s own
# root; it is a second variable rather than a `/tmp` literal because the operator who pointed the
# grade directory somewhere is usually the operator whose `/tmp` is too small for a test suite's
# output.
LOG_DIR_ENV = "V3_LOG_DIR"

# The sandbox-host seam (§8b.9). It names the daemon the sandbox fleet runs —
# `unix:///run/user/1000/docker.sock` for a rootless daemon beside the Orchestrator,
# `ssh://user@host` for a separate machine. UNSET IS NOT A VALUE: nothing is executed until it says
# something (see the module docstring for the measurement that rule exists for).
DOCKER_HOST_ENV = "V3_DOCKER_HOST"

# The single-box declaration: grade on whatever this client resolves by itself, ambient `DOCKER_HOST`
# and `docker context` included — i.e. byte for byte what an unset seam used to do, said on purpose.
# A literal rather than a second variable, so the answer to "which daemon graded" is in one place.
DOCKER_HOST_LOCAL = "local"

# Endpoint schemes that name a daemon on ANOTHER MACHINE — the arrangement where `workspace()`'s
# payload provably cannot be reached, since a tempdir on this filesystem is not on that one. Checked
# by scheme because that is the only part of an endpoint whose meaning is fixed: a host name says
# nothing (`ssh://localhost` is still a second filesystem namespace when the far side is a
# container). `unix://` is deliberately ABSENT — a second daemon on this box shares the filesystem
# and fails, if it fails at all, on the uid instead (the 0700 tempdir), which no environment read can
# answer and `preflight`'s by-content probe can.
REMOTE_ENDPOINT_SCHEMES = ("ssh://", "tcp://", "http://", "https://")

# How many names go into one `docker image inspect`, and the two bounds are opposite. Fewer chunks
# is fewer round trips, which is the whole point over an ssh transport (0.84 s per docker invocation
# multiplexed, 2.73 s without — the sandbox record §1.3a); but one argv holding a whole corpus is a
# command line of half a megabyte against a ~2 MB ARG_MAX, i.e. a census that dies on the biggest
# benchmark and works on the others. 256 names is ~30 kB of argv and 27 round trips for R2E-Gym's
# 6812 tasks.
IMAGE_CENSUS_CHUNK = 256

# What a batched inspect prints: the daemon's own tags, one per line, for each image it resolved.
# `{{println}}` per tag rather than `{{json .RepoTags}}` because the answer is then plain text a
# `.split()` reads — and an image with no tags at all contributes nothing, which is the correct
# answer about it.
IMAGE_TAGS_FORMAT = "{{range .RepoTags}}{{println .}}{{end}}"

# `docker version` against a live daemon answers in milliseconds. The bound is here for the endpoint
# that is merely GONE, where the client waits on a TCP or ssh connect instead — and this probe runs
# inside `image_present`, i.e. inside a corpus check whose whole value is being fast.
DAEMON_PROBE_TIMEOUT_S = 20.0

# How long `preflight` gives the probe container. Generous because it is paid once per deployment and
# a first `docker run` against a cold daemon is not a fast operation; still bounded, because a
# preflight that hangs is worse than the misconfiguration it looks for (`koth/doctor.py`).
PREFLIGHT_TIMEOUT_S = 120.0

# `docker run` exits 125 when DOCKER ITSELF failed to start the container — a missing image under
# `--pull never`, a daemon that is down, a flag this Docker does not know. It is the one exit code
# that is unambiguously the validator's rather than the workload's, so it is the one this module
# converts into a refusal instead of returning.
DOCKER_RUN_FAILED = 125

# ...but not the only code docker's OWN failures arrive on. Measured: a bind source the daemon may
# not create (`error while creating mount source path …: permission denied`) exits 126, and an
# entrypoint the image does not have exits 127 — the CLI reuses the shell's "cannot invoke" and "not
# found" codes for a container it could not start. Both are ALSO legal exits for a program that
# really ran, so widening `DOCKER_RUN_FAILED` to cover them would convert an honest submission into
# an exclusion, which §8b.3 rates as bad as scoring a harness failure. The code is therefore only
# half the discriminator; `_docker_said_it_failed` is the other half.
DOCKER_MAY_HAVE_FAILED = (126, 127)

# The other half: docker's own voice. The CLI prefixes every error it prints with `docker: `, and a
# container that ran writes only its own output to this log — so the pair (one of the codes above,
# and a line docker itself wrote) separates "the container never started" from "the program ran and
# exited 126" without guessing from the number alone. Two honesty notes. (1) It is not forgery-proof
# where the log is the miner's: LiveCodeBench's driver captures the submitted program's output, so
# neither half is reachable there, but a SWE-bench/R2E-Gym patch can make a test suite print this
# line and exit 126 to force its task out of both arms. That oracle already exists in those adapters
# — a patch that suppresses the parser's markers is excluded today — so this widens nothing, and
# closing it needs a marker those graders do not have. (2) A false positive costs one dropped task,
# symmetric across both arms; a false negative scores the validator's own breakage as the miner's
# answer. The asymmetry is why the pair is checked and not just the prefix.
DOCKER_CLI_ERROR = "docker: "

# How much of the log is kept in memory for `SandboxResult.output`. The full log stays on disk at
# `log_path` — SWE-bench's shipped parser reads the file, and a test suite's output routinely runs to
# megabytes, which is exactly the size that should never be materialised into an exception message.
MAX_CAPTURE_BYTES = 1 << 20

# Defaults sized for a self-contained program (LiveCodeBench). A repository test suite needs more and
# says so at its call site; both are HOST-PROTECTION bounds rather than grading inputs, which is why
# breaching one produces no verdict and therefore an exclusion, never a zero.
MEMORY = "1g"
PIDS_LIMIT = 512
CPUS = "2"

# `docker info`'s own report of which of those caps the daemon can actually enforce, mapped to the
# flag each one silently disarms. THIS IS MEASURED, NOT DEFENSIVE: a rootless daemon started without
# a delegated cgroup — a bare `dockerd-rootless.sh`, i.e. no systemd user session and no `loginctl
# enable-linger` — ACCEPTS `--memory`, `--pids-limit` and `--cpus` and ignores them, reporting "No
# memory limit support" as a startup warning nobody is reading at 3am. Those three flags are the only
# thing standing between one graded task and a validator that dies mid-window, so a daemon that
# cannot enforce them is not a sandbox and `preflight` refuses it. `SwapLimit` is deliberately NOT in
# this list: it is false on plenty of otherwise-correct hosts, and losing `--memory-swap` degrades
# the memory cap to RAM-only rather than removing it.
CAP_FIELDS = (("MemoryLimit", "--memory"), ("PidsLimit", "--pids-limit"), ("CPUCfsQuota", "--cpus"))

# How long `_kill` keeps asking for the container to go away. One removal is NOT enough, and the hole
# it leaves is the one this whole file is about: `subprocess.run`'s timeout kills the docker CLIENT,
# and if the daemon had not finished creating the container by then, `docker rm -f` answers "No such
# container" — after which the container starts, unattended, and holds its memory for as long as the
# workload wants. That race widens exactly when it hurts, because container creation is slowest when
# the machine is already loaded. Removing repeatedly over a window costs nothing (removing a
# container that does not exist is a no-op) and closes it.
KILL_WINDOW_SECONDS = 5.0
KILL_INTERVAL_SECONDS = 0.5


class SandboxError(RuntimeError):
    """The grading HARNESS failed, as opposed to the graded submission failing (§8b.3).

    The distinction is the whole reason this class exists rather than a `False` return: a worker
    model erroring is an OUTCOME — information about that rung, which a Conductor that keeps
    delegating there should be scored for — while Docker dying is a fact about the owner's machine.
    Raised so it reaches `scaffold`'s caller, which drops the task from BOTH arms and from the
    denominator (§6.3c); a version of this that returned 0.0 would charge a miner for the validator's
    infrastructure and nothing downstream could tell afterwards.
    """


@dataclass(frozen=True)
class SandboxResult:
    """What the container did: its exit code, and where its output went.

    `exit_code` is data, not a verdict. `git apply` refusing a malformed patch exits non-zero and is
    a fact about the miner's answer; a test runner exiting non-zero because tests failed is the
    ordinary case. Only the adapter's own verdict marker can tell those from a driver that never ran,
    so the decision is left where the marker is.
    """

    exit_code: int
    output: str                 # the tail of the log, capped at MAX_CAPTURE_BYTES
    log_path: pathlib.Path      # the whole log, for a grader that parses it as shipped


@contextlib.contextmanager
def workspace() -> Iterator[tuple[pathlib.Path, pathlib.Path]]:
    """Two throwaway directories, `(payload_root, log_root)`. The daemon can see only the first.

    Both start empty. Callers put the container's payload in a subdirectory of `payload_root` and
    pass `log_root / "*.log"` as `run`'s `log_path` — the same contract as before, that the log is
    never inside the mount, now kept by ALLOCATION rather than by everyone remembering to put the
    file one level up.

    `payload_root` is what `docker run -v` is aimed at, so it is allocated under `V3_GRADE_DIR`:
    the bind is resolved by the DAEMON and the path has to exist on its side too (module docstring).
    `log_root` is under `V3_LOG_DIR`, i.e. under `tempfile`'s own root unless an operator says
    otherwise, and nothing in the container ever reaches it.
    """
    with tempfile.TemporaryDirectory(dir=os.environ.get(GRADE_DIR_ENV) or None) as payload_root, \
            tempfile.TemporaryDirectory(dir=os.environ.get(LOG_DIR_ENV) or None) as log_root:
        yield pathlib.Path(payload_root), pathlib.Path(log_root)


def docker_host() -> str | None:
    """The daemon this module grades on, or None for whatever the client resolves by itself.

    None means two different things to the CALLER of this function and one thing to the command line:
    add no `--host`. Whether that None was declared (`local`) or merely absent is a separate question
    with a separate function, because only one of the two is allowed to execute anything.

    Read at call time rather than at import, like `GRADE_DIR_ENV`: a deployment sets it in the
    process environment, and a test has to be able to ask both questions of the same interpreter.
    """
    host = os.environ.get(DOCKER_HOST_ENV) or None
    return None if host == DOCKER_HOST_LOCAL else host


def _where(host: str | None) -> str:
    """The endpoint as an operator's message names it. One spelling, because two gates print it."""
    return (f"{DOCKER_HOST_ENV}={host}" if host else
            f"the client's default endpoint ({DOCKER_HOST_ENV}={DOCKER_HOST_LOCAL})")


def declared() -> bool:
    """Whether an operator has SAID which daemon grades. Silence is not consent (§8b.9).

    The one bit that separates "this box grades on its own daemon, deliberately" from "nobody has
    configured this yet", which an unset `DOCKER_HOST` cannot express — see the module docstring for
    the measurement that makes the difference a security property rather than a preference.
    """
    return bool(os.environ.get(DOCKER_HOST_ENV))


def _require_declaration() -> None:
    """Refuse before anything is executed, naming both ways to say it. `run` and `preflight` call it.

    A `SandboxError` because that is what the callers already handle: at grading time it is the
    §8b.3 exclusion (the task leaves BOTH arms, uncorrupted), and at preflight time it is exit 1 with
    the fix printed. Nothing here is a fresh exception type, because a misconfiguration and a dead
    daemon want the same treatment — refuse, loudly, before a number exists.
    """
    if declared():
        return
    raise SandboxError(
        f"{DOCKER_HOST_ENV} is unset, so no grading host has been DECLARED and nothing will be "
        "executed. Grading runs a program a miner's model wrote, and the default endpoint on the "
        "box that holds the chain hotkey has been measured to be the root daemon (§8b.9). Say which "
        f"it is: `{DOCKER_HOST_ENV}=unix:///run/user/1000/docker.sock` (or ssh://user@host) for the "
        f"sandbox fleet, or `{DOCKER_HOST_ENV}={DOCKER_HOST_LOCAL}` to state out loud that this box "
        "grades on its own default daemon")


def _docker(*args: str) -> list[str]:
    """A docker command line aimed at the configured daemon.

    Every subprocess in this module is built here, and that is the point: three entry points quietly
    disagreeing about WHICH daemon they mean is the failure mode the seam exists to make impossible —
    an `image_present` answered by the Orchestrator's own daemon is a corpus check that passes for a
    fleet that has none of the images.
    """
    # The gate lives here too, not only in the three callers. This function is documented as the
    # one place every subprocess in this module is built, which makes it the drift-proof
    # chokepoint; every caller already refuses before reaching it, so today this costs nothing and
    # tomorrow a new helper cannot forget.
    _require_declaration()
    host = docker_host()
    return [DOCKER, *(("--host", host) if host else ()), *args]


def available() -> bool:
    """Whether this process MAY grade, and can. False on an undeclared box, before any probe.

    The declaration is checked first and without touching docker, because "can we reach a daemon"
    and "are we allowed to execute a miner's program on it" are different questions and this
    function is asked in only one context — deciding whether the grading path is open at all
    (`image_present`, and the live test suite's `needs_docker` skip). An undeclared box answering
    True here is how the measured failure got its container: the client was on PATH, so everything
    downstream believed grading was configured.

    Cheap once declared `local`: a `docker` client on PATH is the ordinary local-socket case, and a
    local daemon that is down still arrives as a raise from `run` rather than as a wrong answer here.
    When `V3_DOCKER_HOST` names another host or another uid, a client on PATH says nothing about
    that daemon, so this pays for one `docker version` — the alternative is a corpus check that
    passes on a machine that cannot grade.
    """
    if not declared():
        return False
    if shutil.which(DOCKER) is None:
        return False
    if docker_host() is None:
        return True
    try:
        return subprocess.run(_docker("version", "--format", "{{.Server.Version}}"),
                              capture_output=True,
                              timeout=DAEMON_PROBE_TIMEOUT_S).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def image_present(image: str) -> bool:
    """Whether `image` is already on the GRADING daemon, so `--pull never` will not refuse it.

    Provisioning images is an operator step (see the module docstring); this is what lets a window
    check its corpus before it opens rather than discovering a missing image inside a graded episode.
    Asked of the configured daemon and never of the local one: an image present here and absent there
    is the same answer as no image at all, arriving two hours later.
    """
    if not available():
        return False
    return subprocess.run(_docker("image", "inspect", image),
                          capture_output=True).returncode == 0


def images_present(names: Iterable[str]) -> set[str]:
    """Which of `names` the GRADING daemon already has — the corpus census, batched.

    `image_present` asked once per task pays TWO round trips per task, because it calls `available()`
    and `available()` issues a live `docker version` whenever `V3_DOCKER_HOST` names a daemon of its
    own. R2E-Gym's 6812 tasks are therefore ~13,600 invocations, and at the 0.84 s a multiplexed ssh
    `docker` was measured to cost (the sandbox record §1.3a) that is 2000-4000 s — a bill nothing
    pays today only because window building does not yet take the census the three `usable_ids`
    docstrings say it is for. Here the reachability probe is paid ONCE and the inspect once per
    `IMAGE_CENSUS_CHUNK`: 27 round trips for that corpus. The probe itself is not dropped — a client
    on PATH says nothing about a daemon on another host, which is exactly its job; what was redundant
    is paying for it per image.

    PRESENCE IS READ OFF THE RETURNED OBJECTS AND NEVER OFF THE EXIT CODE, which is non-zero when
    ANY name in the chunk is missing. Docker returns one object per argument it resolved, in order,
    silently skipping the rest, so the objects cannot be matched back positionally — a requested name
    is matched against the tags the daemon reports for the images that did come back.

    CONSERVATIVE BY CONSTRUCTION, and that is the direction to be wrong in. A name that is literally
    one of a local image's tags is a name `docker run --pull never` resolves, so there are no false
    positives; a name docker would resolve some OTHER way — a digest, an ID prefix, a
    `docker.io/library/…` spelling that `RepoTags` normalises away — reads as absent here, and so
    does everything if the daemon dies mid-census. That costs a task its place in a window, which is
    what not provisioning it costs anyway; the opposite error is `--pull never` refusing at task 1 of
    a graded arm (§8b.2). `image_present` remains the exact single-name answer and stays the one
    `preflight` asks, because there the question is one image and the cost of a false negative is a
    refusal to launch.
    """
    wanted = sorted(set(names))
    if not wanted or not available():
        return set()
    tags: set[str] = set()
    for start in range(0, len(wanted), IMAGE_CENSUS_CHUNK):
        chunk = wanted[start:start + IMAGE_CENSUS_CHUNK]
        done = subprocess.run(_docker("image", "inspect", "--format", IMAGE_TAGS_FORMAT, *chunk),
                              capture_output=True, text=True)
        tags.update(done.stdout.split())
    return {name for name in wanted if name in tags}


def run(image: str, argv: Sequence[str], *, mount: pathlib.Path, log_path: pathlib.Path,
        timeout: float, memory: str = MEMORY, pids: int = PIDS_LIMIT,
        cpus: str = CPUS) -> SandboxResult:
    """Run `argv` in a throwaway `image` container with `mount` at `/w`, read-only.

    Executes untrusted code, so it is the function the declaration gates: an undeclared box raises
    here BEFORE a command line is built, which is the difference between a deployment that documents
    the §8b.9 split and one that has it.

    Raises `SandboxError` when the harness failed — no grading host declared, no docker client,
    docker unable to start the container, or a container that had to be killed on the wall clock (see
    the module docstring for why the last one is counted here rather than scored as a zero).

    The container is named so it can be killed. `docker run` with a client-side timeout kills the
    CLIENT; the container keeps running, holding its memory and its CPU share for as long as the
    workload wants, which on a validator running ~500 graded episodes per window is how one hung task
    becomes an unusable machine. `--rm` does not help, because it fires when the container exits.
    """
    _require_declaration()
    name = f"v3-{secrets.token_hex(8)}"
    command = _docker(
        "run", "--rm", "--name", name,
        "--pull", "never",              # a missing image is an operator failure, not an arm's cost
        "--network", "none",            # no phoning home, and no fetching the expected outputs
        "--memory", memory, "--memory-swap", memory,   # equal: no swap, so the cap is the real cap
        "--pids-limit", str(pids),
        "--cpus", cpus,
        "--security-opt", "no-new-privileges",
        "-v", f"{mount}:{MOUNT}:ro",
        image, *argv,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as log:
        try:
            # Merged streams, because that is what the graders parse: SWE-bench's shipped
            # `get_logs_eval` looks for its markers in the container's combined output, and splitting
            # them here would interleave differently and lose the ordering it depends on.
            done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        except FileNotFoundError as exc:
            raise SandboxError(f"no `{DOCKER}` client on PATH; the owner's validator grades every "
                               "code benchmark in a container") from exc
        except subprocess.TimeoutExpired as exc:
            _kill(name)
            raise SandboxError(f"container {name} exceeded {timeout:g}s on {image} and was killed; "
                               "no verdict, so this task is excluded from both arms") from exc
    result = SandboxResult(exit_code=done.returncode, output=_tail(log_path), log_path=log_path)
    if result.exit_code == DOCKER_RUN_FAILED or (
            result.exit_code in DOCKER_MAY_HAVE_FAILED and _docker_said_it_failed(result.output)):
        raise SandboxError(f"docker could not start {image}: {result.output.strip()[-400:]}")
    return result


def _docker_said_it_failed(output: str) -> bool:
    """Whether the log holds a line the docker CLI wrote about itself, rather than container output.

    The discriminator for the codes docker shares with real programs (`DOCKER_CLI_ERROR`). A line
    rather than a substring, and the CLI's own prefix rather than "Error response from daemon":
    that phrase is quoted by tools that talk to docker, and a graded program printing one is
    reporting rather than failing.
    """
    return any(line.startswith(DOCKER_CLI_ERROR) for line in output.splitlines())


def _kill(name: str) -> None:
    """Remove the container, repeatedly, until the creation race above cannot have outlasted us.

    Best effort throughout: the container is already unreachable by us, so a failure to remove it is
    one more thing wrong with the validator, and raising a second exception here would hide the
    timeout that brought us in.

    BEST EFFORT HAS TO BE WRITTEN DOWN, NOT MEANT, AND ON A REMOTE DAEMON IT IS THE WHOLE FUNCTION.
    `_docker` prepends `--host` when one is configured, so every removal here is a network operation
    over the same link whose health is exactly what is in question — this is the timeout path. Two
    measured failures, both against `V3_DOCKER_HOST=ssh://…`: an unbounded `rm` on a wedged link
    never returns, and because this runs INSIDE `run`'s `except`, the §8b.3 exclusion is never
    raised and the arm blocks for good (the per-duel clock is checked between tasks, so nothing
    downstream can end it); and a client that is simply gone raises out of here and REPLACES the
    `SandboxError` with its own exception, which is the sentence above happening literally.
    So each removal is bounded by the window this function is already allowed to spend, and both
    failure classes are swallowed — a container we could not remove is a leaked container, which is
    what the caller was reporting anyway.
    """
    deadline = time.monotonic() + KILL_WINDOW_SECONDS
    while True:
        try:
            subprocess.run(_docker("rm", "-f", name), capture_output=True, check=False,
                           timeout=KILL_WINDOW_SECONDS)
        except (OSError, subprocess.SubprocessError):
            pass
        if time.monotonic() >= deadline:
            return
        time.sleep(KILL_INTERVAL_SECONDS)


def check_grading_host() -> str:
    """The LAUNCH gate: refuse an arrangement that cannot grade, and say where containers will land.

    `preflight` proves the whole path and spends a container to do it, so it is an operator step run
    once per deployment. This is the cheap half a long-running process can afford at every start, and
    it exists because of what the two failures cost. A validator that starts with the seam pointed
    somewhere unusable does not fail at launch: it opens the window, pays for every worker call in
    the king's arm and every challenger's, and only then discovers that each task raises a
    `SandboxError` — which §8b.3 correctly turns into an exclusion from BOTH arms and the
    denominator, so the window resolves nothing while having spent everything. One `docker version`
    at startup is the whole price of never doing that.

    Four refusals and no more, because only these are answerable from this process's own
    environment:

      * NOTHING DECLARED (`_require_declaration`) — silence is not consent, and this is the check
        that used to be missing while containers landed on the root daemon;
      * A REMOTE ENDPOINT WITH NO SHARED GRADE DIRECTORY — the arrangement that provably cannot work,
        because `docker run -v` is resolved by the daemon and `workspace()`'s tempdir exists only
        here. It is refused BEFORE any probe, since reachability is not the problem with it;
      * EITHER ALLOCATION ROOT NOT A DIRECTORY *HERE* — one `os.path.isdir` each, and they are in
        scope by this gate's own rule: they are facts about this process's own filesystem. Measured:
        with `V3_GRADE_DIR` naming a path that does not exist, this gate passed, and then every
        `workspace()` raised a bare `FileNotFoundError` — not even a `SandboxError` — which
        `_GraderGuard` catches as broadly as any other, so every task left both arms and the
        denominator while the window spent its whole allowance. That is the failure this gate exists
        to turn into one line at launch, arriving through the half of the seam that was unchecked.
        `V3_LOG_DIR` is checked with it because `workspace()` allocates under both, so a typo in
        the newer variable buys the identical window.

    Then it asks the daemon whether it is there, via `available()`. It deliberately does NOT check
    the mount, the caps or the images: those need a container and an image name, they are
    `preflight`'s, and a gate that half-checked the mount would be worse than one that says it did
    not — the return line names the arrangement, not a guarantee about it.
    """
    _require_declaration()
    host = docker_host()
    grade_dir = os.environ.get(GRADE_DIR_ENV)
    if host is not None and host.startswith(REMOTE_ENDPOINT_SCHEMES) and not grade_dir:
        raise SandboxError(
            f"{DOCKER_HOST_ENV}={host} names a daemon on another machine and {GRADE_DIR_ENV} is "
            "unset, so nothing this process builds could be read by it: `docker run -v` is resolved "
            "by the DAEMON, and the payload is a tempdir that exists only here, so the bind would "
            "mount an empty directory and every task would end with no verdict and leave both arms. "
            f"Point {GRADE_DIR_ENV} at storage both machines see at the IDENTICAL absolute path, "
            "then prove it BY CONTENT with `python -m thirtyspokes.v3.benchmarks.sandbox` run in "
            "this process's environment. This refusal rules out the arrangement that cannot work; "
            "it does not check the one that can")
    if grade_dir and not os.path.isdir(grade_dir):
        raise SandboxError(
            f"{GRADE_DIR_ENV}={grade_dir} is not a directory on this box, so `workspace()` could "
            "not build a payload in it: every read and every grade would raise before a container "
            "existed, each task would leave both arms and the denominator (§8b.3), and the window "
            "would resolve nothing having spent everything. Create it here — and, when "
            f"{DOCKER_HOST_ENV} names another machine, at the IDENTICAL absolute path there too")
    log_dir = os.environ.get(LOG_DIR_ENV)
    if log_dir and not os.path.isdir(log_dir):
        raise SandboxError(
            f"{LOG_DIR_ENV}={log_dir} is not a directory on this box, so `workspace()` could not "
            "allocate the container log's root and every grade would raise before a container "
            f"existed — the same window-eating failure as the line above, through the other half of "
            "the same call. Create it here, on LOCAL storage: unlike the grade directory the daemon "
            "never sees this one, and putting it on a shared mount is the cost this variable exists "
            "to avoid")
    if not available():
        raise SandboxError(
            f"no docker daemon answered at {_where(host)}: either there is no `{DOCKER}` client on "
            "PATH here or that endpoint is not up. A validator that cannot start a container grades "
            "nothing — every task would raise, leave both arms and the denominator (§8b.3), and the "
            "window would still spend the money — so this refuses at launch instead")
    return (f"grading lands on {_where(host)}"
            + (f" via {GRADE_DIR_ENV}={grade_dir}" if grade_dir else f" ({GRADE_DIR_ENV} unset)"))


def preflight(image: str, *, timeout: float = PREFLIGHT_TIMEOUT_S) -> str:
    """Prove, before a window opens, that the configured daemon can actually grade. RAISES if not.

    Returns one line describing what was proven, for an operator's log. Everything else is a
    `SandboxError` naming the fix, because a seam that cannot be honoured must refuse rather than
    return a plausible number — §8b.3's rule, applied where it is still free.

    Five failures, each with its own cost and its own repair:
      * no grading host declared — the one checked FIRST and the one that costs nothing to check,
        because the answer this gate used to give on an undeclared box was "exit 0, all good" while
        the containers were landing on the root daemon (module docstring);
      * no client, or a daemon that does not answer at `V3_DOCKER_HOST` — otherwise every graded
        task raises and the entire window is excluded from both arms;
      * a daemon whose cgroup is not delegated, which ACCEPTS `run`'s resource caps and ignores
        them (`CAP_FIELDS`) — the failure that leaves a sandbox looking exactly like a sandbox;
      * `image` absent from THAT daemon — `--pull never` makes it a per-task refusal, and an image
        present on the Orchestrator's own daemon proves nothing at all about the fleet's;
      * the bind mount resolving to something other than our directory — the testnet 526 bug, which
        a second daemon promotes from an exception to the default (two daemons share a path only if
        they share a filesystem, and a rootless one runs as a uid that may not be able to read the
        0700 directory `workspace()` just made).

    The mount is checked BY CONTENT, with a token minted here and never reused: identical path
    strings are what the measured failure already looked like from this side, so a check that
    compared them would have passed on the day it was needed. And it is checked through `run`
    itself — the real `--pull never`, `--network none`, `-v` command line — so what passes here is
    the command grading issues rather than a simpler one that happens to work.
    """
    _require_declaration()
    host = docker_host()
    where = _where(host)
    if shutil.which(DOCKER) is None:
        raise SandboxError(f"no `{DOCKER}` client on PATH; the sandbox host is reached with a client "
                           "here and a daemon there, so this box needs the CLI package too")
    try:
        version = subprocess.run(_docker("version", "--format", "{{.Server.Version}}"),
                                 capture_output=True, text=True, timeout=DAEMON_PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise SandboxError(f"no answer in {DAEMON_PROBE_TIMEOUT_S:g}s from the docker daemon at "
                           f"{where} — an endpoint that hangs is not a grading host") from exc
    if version.returncode != 0:
        # The HEAD of this one, unlike `_tail`: a client error leads with what it could not do
        # ("Cannot connect to the Docker daemon at …") and trails into the dialler's own detail.
        raise SandboxError(f"no docker daemon at {where}: {version.stderr.strip()[:200]} — point "
                           f"{DOCKER_HOST_ENV} at the sandbox fleet's endpoint (a rootless daemon's "
                           "socket, or ssh://user@host) and make sure it is running")
    ignored = _caps_ignored()
    if ignored:
        raise SandboxError(
            f"the daemon at {where} would ACCEPT {' '.join(ignored)} and ignore them: it has no "
            "delegated cgroup, so `docker info` reports no limit support and every cap `run` passes "
            "silently vanishes. One graded task can then take the whole validator down mid-window. "
            "For a rootless daemon the fix is a systemd user session that survives logout — "
            "`loginctl enable-linger <user>`, then restart the daemon through `systemctl --user` "
            "rather than as a bare dockerd-rootless.sh")
    if not image_present(image):
        pull = f"docker {f'--host {host} ' if host else ''}pull {image}"
        raise SandboxError(f"{image} is not on the daemon at {where}, and `run` passes --pull never, "
                           f"so every task needing it would refuse mid-window: `{pull}`")
    token = secrets.token_hex(8)
    probe = "probe.txt"
    with workspace() as (work, logs):
        payload = work / "payload"
        payload.mkdir()
        (payload / probe).write_text(token)
        try:
            result = run(image, ["cat", f"{MOUNT}/{probe}"], mount=payload,
                         log_path=logs / "preflight.log", timeout=timeout)
        except SandboxError as exc:
            raise SandboxError(f"the probe container did not run on the daemon at {where}: {exc}") \
                from exc
        if result.exit_code != 0 or result.output.strip() != token:
            raise SandboxError(
                f"the daemon at {where} answers, but {payload} did not arrive at {MOUNT}: expected "
                f"{token}, got {result.output.strip()[-200:]!r} (exit {result.exit_code}). "
                "`docker run -v` is resolved by the DAEMON, so that path must exist on ITS side "
                f"with these contents and be readable by the uid it runs as. Set {GRADE_DIR_ENV} to "
                "a directory both sides see at the IDENTICAL path (and, for a rootless daemon, one "
                "its user can read); grading is impossible until this probe passes")
    grade_dir = os.environ.get(GRADE_DIR_ENV)
    return (f"docker {version.stdout.strip()} at {where}; caps enforced; {image} present; {MOUNT} "
            "bind resolves"
            + (f" via {GRADE_DIR_ENV}={grade_dir}" if grade_dir else f" ({GRADE_DIR_ENV} unset)"))


def _caps_ignored() -> tuple[str, ...]:
    """Which of `run`'s resource caps this daemon would accept and silently drop (`CAP_FIELDS`).

    Asked of `docker info`, which is the daemon's own answer about the controllers it holds, rather
    than inferred from the endpoint looking rootless: a rootless daemon WITH a delegated cgroup
    enforces all three (measured: an allocation loop OOM-killed at `--memory 1g`, thread creation
    refused at 511 under `--pids-limit 512`), so the shape of the socket path predicts nothing.

    Fails closed on an answer it cannot read. The alternative is treating "I could not tell" as "the
    caps are fine", which is the exact shape of the bug this exists for — and the cost of being
    wrong is one loud preflight against a docker old enough to lack these fields, versus a window of
    uncapped containers on the machine that sets weights.
    """
    fields = " ".join("{{." + field + "}}" for field, _ in CAP_FIELDS)
    try:
        info = subprocess.run(_docker("info", "--format", fields), capture_output=True, text=True,
                              timeout=DAEMON_PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise SandboxError(f"no answer in {DAEMON_PROBE_TIMEOUT_S:g}s from `docker info`, so "
                           "whether this daemon enforces the sandbox's resource caps is "
                           "unknown") from exc
    answers = info.stdout.split()
    if info.returncode != 0 or len(answers) != len(CAP_FIELDS) or any(
            answer not in ("true", "false") for answer in answers):
        raise SandboxError(
            "could not read cap support from `docker info "
            f"--format '{fields}'` (exit {info.returncode}): {(info.stdout + info.stderr)[:200]!r}. "
            "A sandbox whose caps cannot be verified is not a sandbox, so this refuses rather than "
            "assuming they hold")
    return tuple(flag for (_, flag), answer in zip(CAP_FIELDS, answers) if answer == "false")


def main() -> None:
    """`python -m thirtyspokes.v3.benchmarks.sandbox` — run it ON THE BOX THAT WILL GRADE.

    Not on the box that merely orchestrates: every question it asks — the daemon's caps, the uid the
    bind mount must be readable by, the images provisioned — is about the machine whose docker
    executes the miner's program, and it must be asked with the same environment grading will have.

    Not wired into the validator's own startup: this one spends a container, and §8b.9's question is
    about the deployment rather than about the process, so it is answered when the deployment changes
    and not on every restart.
    """
    import argparse

    # LiveCodeBench's image, imported from the adapter that pins it rather than copied: the probe has
    # to use an image the daemon will really be asked for, and this is the one every deployment
    # provisions. `real` imports this module, so the import is here and not at the top.
    from .real import LCB_IMAGE

    parser = argparse.ArgumentParser(
        description="preflight the v3 grading sandbox: a grading host declared, its daemon "
                    "reachable, its resource caps really enforced, the image present, and the bind "
                    "mount resolving identically on both sides")
    parser.add_argument("--image", default=LCB_IMAGE,
                        help="an image the grading daemon must already have (default: %(default)s)")
    args = parser.parse_args()
    try:
        print(preflight(args.image))
    except SandboxError as exc:
        raise SystemExit(f"sandbox preflight FAILED: {exc}")


def _tail(log_path: pathlib.Path) -> str:
    """The last `MAX_CAPTURE_BYTES` of the log, decoded leniently.

    The tail rather than the head: a driver prints its verdict when it finishes, and an error message
    built from the first megabyte of a test suite's output would name the wrong thing. Lenient
    decoding because a graded program's stdout is arbitrary bytes and a UnicodeDecodeError here would
    turn a scoreable submission into an exclusion.
    """
    size = log_path.stat().st_size
    with open(log_path, "rb") as handle:
        if size > MAX_CAPTURE_BYTES:
            handle.seek(size - MAX_CAPTURE_BYTES)
        return handle.read().decode("utf-8", errors="replace")


if __name__ == "__main__":
    main()
