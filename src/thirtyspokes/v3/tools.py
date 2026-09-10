"""The worker read channel — three read-only verbs a WORKER may issue (§2.1, the sandbox record §7).

THE MEASUREMENT THIS MODULE EXISTS FOR, and it is one sentence:

    0 of 75 R2E-Gym episodes produced a patch `git apply` would accept, while the benchmark's OWN
    fix scored exactly 1.0 on 24 of 25 of those same tasks through the identical grader.

3 models x 25 tasks, a 30x price ladder, every model 0.0000, routable band **+0.0000** — a
zero-variance stratum that costs 53.9 GB of images and resolves nothing. Grading wall clock on the
scored episodes was median 0.2 s against a 0.177 s bare container start: pytest never ran. Every
answer was *syntactically* a unified diff (25/25 well-formed) and every one was refused at the apply
step, with an invented blob hash (`1234567..abcdefg`), invented line numbers and invented context
lines — necessarily invented, because the worker has never seen the file. `git apply` matches CONTEXT
LINES, so the single missing input is the bytes of the file, and reading is sufficient to supply it.
The gold control is what makes that attributable: the harness is sound, the failure is the ACTION
SPACE.

WHY READING AND NOTHING ELSE. Writing, applying, running tests and a shell are each unnecessary to
remove the measured failure, and each costs three things: attack surface against §2.1, dollars out of
the miner's allowance, and — the one that is not recoverable — the property that a read against a
pinned image is a deterministic, re-executable function a third party can replay. A test runner also
reaches the graded suite, and the sandbox record §7.1 measured that hazard exactly: a network-dependent
test that `--network none` turns into a SKIP zeroes the GOLD patch too.

THE WORKER ISSUES THESE CALLS. NEVER THE CONDUCTOR (§2.1). If the Conductor issued them its output
would decide *which bytes appear in a worker's context*, which is "influencing WHAT the worker is
asked" in the one form §2.1 exists to forbid. Four grounds, the last of which survives every
mitigation:

  * it needs a free-text argument, and `types.Action` exists in order to have none — all twelve
    `ATTACKS` in `tests/test_invariant.py` die on the fact that a solution cannot be smuggled
    through a type that cannot carry one;
  * the error path is a direct byte channel: a path that does not exist must come back quoting the
    path, or the caller cannot recover, so arbitrary Conductor bytes land verbatim in a worker
    prompt;
  * even a whitelist-validated path is a channel, because the SELECTION carries the bits — choosing
    among ~4,000 files is ~12 bits per step over which region of the repository the worker sees, and
    localisation is precisely the input the measurement says is missing. A Conductor that memorised
    the gold patch would be scored for knowing the benchmark rather than for routing;
  * Conductor tool turns would be unpriced. A worker turn is a provider call, metered by the gateway
    and landing in `spend_usd`, so the loop pays for itself out of the miner's own allowance and
    `final_b = quality_b - λ_b · spend_b / C_b` charges a policy that burns turns.

THE TRANSCRIPT IS DELEGATE-SCOPED, NOT EPISODE-SCOPED, and that is enforced by SCOPE rather than by a
rule: it is a local variable born and dying inside one `_call_worker` call. Under episode scoping,
whether the strong model sees the cheap model's reads would be decided by the Conductor's choice of
when to RETRY — no Conductor *bytes* would flow, but the CONTENT of a worker's prompt would become a
function of a Conductor decision, and that is the difference between an invariant that is enforced
and one that is argued. It also opens a cross-model laundering path (a miner-controlled cheap model
emits `FIND <memorised patch>`, a frontier model reads it back) and it would measure the retried
model on a different input than the first, which is not the comparison §3 wants. The price is wall
clock, and `config.py` states in as many words that the wall clocks are not one-way doors. §2.1 is
not a knob.

CONFINEMENT IS THE WHOLE SECURITY STORY, IN THREE LAYERS, because a miner can route to a model they
control and every tool the worker can call is therefore a tool a miner can drive. There is a specific
prize: **`/r2e_tests` sits OUTSIDE `/testbed`**, and `r2egym.py` says in as many words that this is
what makes the benchmark safe to grade — an unconfined read hands a miner the graded test file, and a
patch that special-cases it scores 1.0 without fixing anything.

  1. `parse_tool` refuses an argument that is empty, over `MAX_ARG_BYTES`, or carries a NUL or a line
     break; `types.ToolCall` refuses the last two again at construction.
  2. `observe` refuses an absolute path or one with a `..` component BEFORE a container is built.
  3. `READER` re-resolves the path with `os.path.realpath` INSIDE the container and refuses anything
     that is not the root or under it — after symlink resolution, which is the only place a symlink
     to `/r2e_tests` can be caught — and re-checks every file `FIND` walks to.

Root confinement alone excludes `/r2e_tests`, because it is outside `/testbed`. That is why the root
is the repository and not `/`, and it is tested adversarially rather than reviewed.

THE ONE PRIZE ROOT CONFINEMENT DOES NOT REACH IS INSIDE THE ROOT, and that is `HIDDEN` below.

THE RESIDUAL, MEASURED RATHER THAN ARGUED: a HARD link inside the root to a file outside it is
readable, because a hard link has no path to resolve — `realpath` cannot tell it from the original
and neither can any path-based check. It is out of a miner's reach (the images are the owner's pin,
not the miner's build) and it is recorded here rather than defended against, because the only
defences are a device/inode census of the whole tree per read or refusing every file with
`st_nlink > 1`, and the second refuses honest files.

A SECOND MEASURED RESIDUAL, SAME SHAPE: `find()` decodes every file it walks, binaries included, one
LINE at a time. The longest line in the seven images measured is 50.4 MB (a Qt shared object), ~100
MB resident against `--memory 1g` — survivable with ~5x headroom — but an image shipping a ~300 MB
newline-free blob would be OOM-killed at exit 137, which is neither `DOCKER_RUN_FAILED` nor in
`DOCKER_MAY_HAVE_FAILED`, so it arrives as a missing fence and a `SandboxError`: the task leaves both
arms, correctly, and quietly. Also out of a miner's reach for the same reason, and also not defended
against, because the fix is a per-line byte cap in the reader and no image needs one today.

THERE IS NO SECOND EXECUTION PATH. A read is stateless, so it needs no container lifecycle: one
`sandbox.run` per call with the flags that module already passes — `--rm`, `--pull never`,
`--network none`, the memory/pids/cpu caps, `no-new-privileges`, the read-only mount, and
`V3_DOCKER_HOST` enforced before anything is executed. `benchmarks/sandbox.py` is zero-diff, and
that is the point of reusing the seam. `argv` is `["python3", reader]` with the call passed in a JSON
FILE: there is no shell anywhere, so the worker's argument is never a token on a command line.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping

from .config import (
    FIND_MAX,
    LIST_MAX,
    MAX_READS_PER_DELEGATE,
    OBSERVATION_BYTES,
    READ_LINES,
    TOOL_TIMEOUT_SECONDS,
)
from .types import TOOL_NAMES, Environment, Observation, ToolCall

# A grammar bound, so it lives beside the parser that enforces it rather than in `config.py` with the
# mechanism pins. Long enough for any real path in these repositories and for a search string; short
# enough that `FIND <2 KB of memorised gold patch>` is not a tool call at all — it fails to parse, so
# the reply is the submission and is graded, which is exactly what it would have been anyway.
MAX_ARG_BYTES = 512

# The reader's payload delimiters. `sandbox.run` merges stdout and stderr into one log, so the
# observation has to be fenced: ABSENCE OF THE MARKERS IS A SandboxError, never an empty observation
# — the same rule `r2egym.r2e_parse_log` applies to a missing pytest summary block (§8b.3).
OBS_BEGIN = "<<<v3-OBSERVATION>>>"
OBS_END = "<<<v3-OBSERVATION-END>>>"

# Emitted by the reader AFTER the closing marker when it stopped before exhausting the source.
# Truncation is MARKED, never silent: a replay must be able to tell a whole file from the first 200
# lines of one, and silence would be a lie about what the worker was shown.
#
# OUT OF BAND, AND MEASURED BEFORE IT WAS: inside the fence it was a signal the SOURCE could forge.
# A file whose last shown line was `[truncated]` came back with that line silently deleted and the
# worker told the read had been cut — `READ t.py` over `x = 1\n[truncated]\n` returned `1\tx = 1` and
# `truncated=True`. In-band signalling cannot be made safe here, because the payload is arbitrary
# file bytes; after the marker it is a region only the reader writes.
TRUNCATED = "[truncated]"

# Directories the read channel never exposes, whatever root a benchmark names. `.git` is INSIDE the
# confinement boundary by construction — `/testbed` IS the git repository (`r2egym.py` says so twice)
# — and it is history rather than source: a clone that kept its upstream refs holds the very commit
# the task asks the worker to re-derive, and the reader cannot verify what a third party's image
# contains (`r2egym.py`: the environments are "one third party's Docker Hub account under no pin").
# So it is excluded rather than trusted. Nothing the measurement asked for is lost — the missing
# input was the bytes of the SOURCE file — and pruning it also removes a measured 9% to 162% of
# `FIND`'s tree walk (sympy 3.72 s -> 1.42 s: its `.git` is 204 MB against 18 MB of source).
#
# THE LEVER THAT IS NOT SPENT HERE, recorded because it is the largest one and it is not this
# module's to spend: 87-99% of every tree FIND walks is a vendored `/testbed/.venv`, not source.
# Hiding it too would take the corpus's worst walk from 18.4 s to ~0.7 s and remove the `find()`
# residual below entirely — but it changes WHAT A WORKER CAN SEE (a bug in an installed dependency
# stops being findable), which is an owner decision about the action space and not a wall-clock fix.
HIDDEN = (".git",)

_SEPARATOR = "\n\n--- INSPECTION RESULTS ---\n"
_FOOTER = ("\n--- END OF INSPECTION RESULTS ---\n"
           "You may inspect {remaining} more time(s). Reply with one inspection line, or with your "
           "final answer.\n")
# THE LAST TURN NEEDS ITS OWN SENTENCE, and the general one was actively harmful there. At the cap
# `compose` still emitted "You may inspect 0 more time(s). Reply with one inspection line, or with
# your final answer." -- inviting exactly the move `_read` no longer honours, so the tool line
# became the submission and was graded as a patch. That is the failure this footer exists to
# prevent, reintroduced by the footer itself on the one turn it mattered most.
_FOOTER_LAST = ("\n--- END OF INSPECTION RESULTS ---\n"
                "You have no inspections left. Reply with your final answer.\n")

# The grammar text an adapter appends to ITS OWN task statement (`r2egym.r2e_prompt`), which is what
# keeps `WorkerRequest.text == task.prompt` byte-true on the first turn of every delegate — the
# protocol is INSIDE `task.prompt`. One constant in one place: a per-adapter copy is a per-adapter
# drift, and this string enters the on-chain window commitment (`window.py` hashes `t.prompt` per
# task), so the protocol a worker was shown is committed with the corpus rather than living in a
# module the owner could quietly edit mid-flight.
PROTOCOL = (
    "\n\n"
    "You may inspect the repository before answering. To inspect, make the LAST line of your reply\n"
    "exactly one of:\n"
    f"  READ <path> [start_line]   up to {READ_LINES} numbered lines of one file, from start_line\n"
    f"  LIST <path>                the entries of one directory, sorted (up to {LIST_MAX})\n"
    f"  FIND <text>                up to {FIND_MAX} `path:line:text` matches for a FIXED string\n"
    "Paths are relative to the repository root, and `.` is the root. The result is appended to this\n"
    f"message and you are asked again; you may inspect at most {MAX_READS_PER_DELEGATE} times. Any\n"
    "reply whose last line is not one of those three is taken as your final answer.\n")

# The three verbs' shape. `[0-9]+` rather than `str.isdigit`, which accepts Arabic-Indic and other
# Unicode digits: a model ID is refused for a homoglyph (`tests/test_invariant.py`) and a line
# number is held to the same standard.
_DECIMAL = re.compile(r"[0-9]+")

# --- the reader: OWNER CODE, mounted read-only, run with no shell ---------------------------------
# Written into the payload directory at each call exactly as `r2egym._r2e_runner()` writes `run.sh`.
# It takes its call from a JSON FILE named in argv, never from a command line, so nothing a worker
# wrote is ever a shell token. Every bound is passed in that file rather than baked in here, so
# `config.py` stays the single place the caps are pinned.
#
# DETERMINISM IS THE PROPERTY THE WHOLE AUDIT RESTS ON: listings sorted, `os.walk` ordered by sorting
# `dirs` and `files` in place, hits ordered by (path, line), names only — no sizes, no timestamps, no
# inode data. That is what keeps an observation re-executable against the pinned image, and it is
# what dies the instant a write tool, a shell or a test runner is admitted.
_READER_BODY = '''
import json
import os
import sys


def inside(root, path):
    """`path` resolved through every symlink, and only if it lands at or under `root`."""
    target = os.path.realpath(path)
    return target if target == root or target.startswith(root + os.sep) else None


def metadata(root, target, names):
    """Whether `target` sits in or under a hidden directory — checked on the REALPATH, so a
    symlink named `src` pointing at `.git` is caught where the name alone would miss it."""
    return any(part in names for part in os.path.relpath(target, root).split(os.sep))


def keep(lines, whole, budget):
    """Append `whole`, cut to the budget so that one minified line cannot produce a log the
    sandbox's own tail would then cut the markers off. True if anything was lost."""
    lines.append(whole[:budget])
    return len(whole) > budget


def read(path, start, limit, budget):
    lines, size, cut = [], 0, False
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for number, text in enumerate(handle, 1):
            if number < start:
                continue
            if len(lines) >= limit or size >= budget:
                return lines, True
            cut = keep(lines, "%6d\\t%s" % (number, text.rstrip("\\n")), budget) or cut
            size += len(lines[-1]) + 1
    return lines, cut


def listing(path, limit, hidden):
    # HIDDEN entries are omitted from the listing, not merely refused on use. `READ .git/...` and
    # `LIST .git` already raise and `FIND` already prunes, so a name shown here can never be acted
    # on -- it can only cost the worker one of its `MAX_READS_PER_DELEGATE` reads to discover that.
    # Hiding it removes a trap without removing any capability. (A prior pass left it visible,
    # reading it as an action-space change for the owner to make; it is the same `hidden` tuple, so
    # reverting is one argument.)
    entries = sorted(e for e in os.listdir(path) if e not in hidden)
    return entries[:limit], len(entries) > limit


def find(root, pattern, limit, budget, hidden):
    hits, size, cut = [], 0, False
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in hidden)
        files.sort()
        for name in files:
            path = os.path.join(base, name)
            target = inside(root, path)
            # BOTH layers, on the realpath `inside` already resolved. Testing only for None
            # refused a symlink OUT of the tree but served one that resolved INTO `.git`: READ
            # of that same path raised "repository metadata, not source" while FIND returned its
            # contents. One rule with two implementations, and this was the one missing it.
            if target is None or metadata(root, target, hidden):
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    for number, text in enumerate(handle, 1):
                        if pattern not in text:
                            continue
                        if len(hits) >= limit or size >= budget:
                            return hits, True
                        cut = keep(hits, "%s:%d:%s" % (os.path.relpath(path, root), number,
                                                       text.rstrip("\\n")), budget) or cut
                        size += len(hits[-1]) + 1
            except OSError:
                continue                      # unreadable is not an error the worker can act on
    return hits, cut


def answer(call):
    root = os.path.realpath(call["root"])
    if call["name"] == "FIND":
        return find(root, call["arg"], call["find_max"], call["max_bytes"], call["hidden"])
    target = inside(root, os.path.join(root, call["arg"]))
    if target is None:
        return ["error: %s is outside the repository root" % call["arg"]], False
    if metadata(root, target, call["hidden"]):
        return ["error: %s is repository metadata, not source" % call["arg"]], False
    if call["name"] == "LIST":
        if not os.path.isdir(target):
            return ["error: %s is not a directory" % call["arg"]], False
        return listing(target, call["list_max"], call["hidden"])
    if not os.path.isfile(target):
        return ["error: %s is not a file" % call["arg"]], False
    lines, more = read(target, call["start"], call["read_lines"], call["max_bytes"])
    if not lines:
        return ["error: %s has no line %d" % (call["arg"], call["start"])], False
    return lines, more


with open(sys.argv[1]) as handle:
    CALL = json.load(handle)
try:
    LINES, MORE = answer(CALL)
except OSError as exc:
    LINES, MORE = ["error: %s" % exc], False
print(BEGIN)
print("\\n".join(LINES))
print(END)
if MORE:
    print(TRUNCATED)
'''

READER = (f"BEGIN = {OBS_BEGIN!r}\nEND = {OBS_END!r}\nTRUNCATED = {TRUNCATED!r}\n" + _READER_BODY)


def parse_tool(raw: str) -> ToolCall | None:
    """The reply's LAST non-blank line as a read, or None — meaning `raw` IS the submission.

    INVERTED RELATIVE TO `action.parse`, WHERE THE LAST LINE *MUST* BE AN ACTION, and deliberately:
    a Conductor that cannot emit an action is not routing, whereas a worker that ignores this channel
    must not be punished for ignoring it. So anything that is not exactly one of the three verbs
    means the whole reply is the answer, which is what keeps the no-tools path byte-identical.

    THE ONE KNOWN MISFIRE, recorded rather than hidden: a unified diff whose last context line reads
    `READ something` is consumed as a tool call. The cost is bounded and is never a wrong grade — the
    loop re-asks and the final turn always submits whatever comes back, so the worst case is
    `MAX_READS_PER_DELEGATE` wasted turns on a patch that is then still graded.
    """
    lines = [line for line in (raw or "").splitlines() if line.strip()]
    if not lines:
        return None
    name, _, rest = lines[-1].strip().partition(" ")
    if name not in TOOL_NAMES:
        return None
    if name == "FIND":
        # The pattern is the whole rest of the line, because a fixed search string may hold spaces.
        arg, start = rest.strip(), 1
    else:
        # A path may not. Two fields for LIST and two or three for READ, exactly — so `READ x -3`
        # and `READ x ٣` do not parse rather than becoming a path with a space in it, and the
        # grammar has one reading per line instead of two.
        fields = rest.split()
        if len(fields) == 2 and name == "READ" and _DECIMAL.fullmatch(fields[1]):
            arg, start = fields[0], int(fields[1])
        elif len(fields) == 1:
            arg, start = fields[0], 1
        else:
            return None
    if len(arg.encode()) > MAX_ARG_BYTES:
        return None
    try:
        return ToolCall(name=name, arg=arg, start=start)
    except ValueError:
        # An empty argument, a NUL, a line separator, or `READ <path> 0`. Not a tool call, so the
        # reply is a submission — never a repaired call, which would be the scaffold guessing.
        return None


def _footer(remaining: int) -> str:
    """The closing sentence, which must not offer a read the worker cannot make (see `_FOOTER`)."""
    return _FOOTER.format(remaining=remaining) if remaining > 0 else _FOOTER_LAST


def compose(task_prompt: str, observations: tuple[Observation, ...]) -> str:
    """The worker's text for the next turn. PURE and PINNED — `compose(p, ()) == p`, byte for byte.

    THIS FUNCTION IS THE §2.1 AUDIT. The sentence over every row of every window becomes

        request.text == tools.compose(task.prompt, request.observations)

    which is a RECOMPUTATION by anyone holding the published benchmark and the record, not a claim
    about source code — and on `observations == ()` it is today's sentence unchanged. Two arguments,
    one of which is the benchmark's own statement and the other of which can only have come from
    `observe`; there is no third, so no other byte can be in the result.

    The remaining-reads count is DERIVED from `len(observations)` and the pin, never passed: a field
    that duplicates the transcript is a field that can contradict it (`render.ConductorState` makes
    the same choice about spend). The footer is not cosmetic — without it a worker spends its last
    turn on a tool call that becomes the submission and is graded as a patch, i.e. scored 0 for not
    knowing the budget.
    """
    if not observations:
        return task_prompt
    blocks = [f"$ {_call_line(obs.call)}\n{obs.output}"
              + (f"\n{TRUNCATED}" if obs.truncated else "")
              for obs in observations]
    return (task_prompt + _SEPARATOR + "\n\n".join(blocks)
            + _footer(MAX_READS_PER_DELEGATE - len(observations)))


def observe(env: Environment, call: ToolCall, response_sha256: str, *,
            execute: Callable[[Environment, Mapping[str, object]], str] | None = None
            ) -> Observation:
    """Execute `call` against `env`. THE ONLY CONSTRUCTOR OF `Observation` IN THE SYSTEM.

    That is the same move `types.Action` makes on the Conductor side: a channel that cannot be
    spelled. Its inputs are an `Environment` the BENCHMARK named and a `ToolCall` that `parse_tool`
    returned from a WORKER reply, and no function anywhere accepts a caller-supplied observation,
    output or transcript string — so a Conductor byte has nowhere to enter.

    THE TWO FAILURE CLASSES ARE KEPT ON OPPOSITE SIDES OF §8b.3, mirroring `r2egym.py`'s
    `R2E_APPLY_FAIL`-versus-no-summary-block discrimination exactly. A call the WORKER got wrong —
    a path outside the root, a file that does not exist, a `start` past EOF — comes back as an
    `Observation` carrying an error string: it costs a turn, it is deterministic, and it is feedback.
    A container that FAILED — image absent under `--pull never`, daemon down, no markers, killed on
    the clock — raises `sandbox.SandboxError`, which reaches the caller and drops the task from BOTH
    arms and from the denominator (§6.3c).

    `execute` is the seam the offline suite replaces (`benchmarks.mock.in_process`), for the reason
    `Scaffold.clock` is one: the whole loop is then testable with no Docker, against the SAME reader
    source the container runs.
    """
    refusal = _refused_path(call)
    if refusal is not None:
        return Observation(call=call, response_sha256=response_sha256, output=refusal,
                           truncated=False)
    log = (execute or in_sandbox)(env, _request(call, env))
    payload, truncated = _payload(log)
    return Observation(call=call, response_sha256=response_sha256,
                       output=payload[:OBSERVATION_BYTES],
                       # Either the reader stopped early or the container returned more than the pin
                       # allows. The second is a defence rather than an expectation: the reader is
                       # ours, but the interpreter running it belongs to the benchmark's image.
                       truncated=truncated or len(payload) > OBSERVATION_BYTES)


def in_sandbox(env: Environment, request: Mapping[str, object]) -> str:
    """One read, in a throwaway container on the benchmark's own per-task image. Returns its log.

    The default `execute`. Every flag is `sandbox.run`'s own — including `V3_DOCKER_HOST`, which it
    refuses to execute without (§8b.9) — because reusing that seam rather than opening a second
    execution path is the point: a read tool on the box that holds the chain hotkey is a hotkey
    reading tool, and the split between the Orchestrator and the sandbox fleet is what stops it.

    Measured end to end on seven of these images (the sandbox record §7.4): container start is
    0.26-0.32 s ON THE SANDBOX HOST, against the 0.177 s this docstring used to quote from §7 — and
    that gap is the BOX, not the flag set. Checked back to back on the sandbox host: §7's bare
    command line (no mount, no caps) costs 0.244 s median and this function's full command line
    costs 0.249 s, n = 7 each, so the caps and the read-only bind are free and §7's number is not an
    undercount of the same thing — it was timed on the controller's daemon, which is the box §8b.9
    took grading away from. 0.26-0.32 s is therefore the figure to size against, because the machine
    that will start these containers is the sandbox host. The harness's own overhead above
    `sandbox.run` is 0.67 ms, so the tool's cost IS the container's.
    LIST and READ are that start and nothing else, 0.26-0.37 s whatever the tree. FIND is the one
    verb with a workload: a linear walk at ~160 MB/s, 1.3 s median and 18.4 s on the corpus's largest
    repository cold, which is what `TOOL_TIMEOUT_SECONDS` is pinned against.
    """
    # Local because `benchmarks/__init__` imports the adapters and `r2egym` imports this module: the
    # pure half of this file — the grammar and the composition — touches no execution seam at all.
    from .benchmarks import sandbox

    with sandbox.workspace() as (work, logs):
        payload = work / "payload"
        payload.mkdir()
        (payload / "reader.py").write_text(READER)
        (payload / "call.json").write_text(json.dumps(request, sort_keys=True))
        result = sandbox.run(env.image,
                             # `-I` is isolated mode, and it is NOT cosmetic. Measured: an image
                             # that sets PYTHONPATH can substitute the `json` module that runs
                             # INSIDE this reader -- the owner code enforcing realpath containment
                             # and HIDDEN. These images are third-party and `r2egym.py` says in as
                             # many words that they are under no pin, so the image's environment
                             # must not choose what owner code imports. `-I` implies `-E` (ignore
                             # PYTHONPATH/PYTHONHOME) and `-s` (no user site). Cost: 0 ms.
                             ["python3", "-I", f"{sandbox.MOUNT}/reader.py",
                              f"{sandbox.MOUNT}/call.json"],
                             mount=payload, log_path=logs / "read.log",
                             timeout=TOOL_TIMEOUT_SECONDS)
        # `run` already tailed this file into `result.output`. Re-reading it is free -- 0.03 s an
        # arm, and since the log moved off V3_GRADE_DIR (`sandbox.workspace` yields its root
        # separately) it is local even under the shared-mount arrangement, where these were 2-4 of
        # the ~30 round trips a read used to cost at 1652 ms against 0.24 ms.
        # Use the tail when it carries the fence; fall back otherwise, because `_tail`
        # keeps only the last MAX_CAPTURE_BYTES while `_payload` looks for the FIRST `OBS_BEGIN`,
        # and a pathological log could put them on opposite sides of that cut. The reader caps
        # itself at OBSERVATION_BYTES, so the fast path is every real case and the fallback is
        # the argument rather than the behaviour.
        if OBS_BEGIN in result.output:
            return result.output
        return result.log_path.read_text(errors="replace")


def _request(call: ToolCall, env: Environment) -> dict[str, object]:
    """The reader's whole input. Every cap is passed rather than baked into `READER`, so the pins
    stay in `config.py` and a replay can read what the reader was actually bounded by."""
    return {"name": call.name, "arg": call.arg, "start": call.start, "root": env.root,
            "read_lines": READ_LINES, "list_max": LIST_MAX, "find_max": FIND_MAX,
            "max_bytes": OBSERVATION_BYTES, "hidden": list(HIDDEN)}


def _refused_path(call: ToolCall) -> str | None:
    """Confinement layer 2: refused BEFORE a container is built, so the read costs nothing.

    `FIND` takes a fixed search string rather than a path and is not checked here — its confinement
    is the walk's root, which is layer 3's business. The message quotes the worker's own argument,
    which is safe for exactly the reason a Conductor-issued call would not be: these bytes came from
    the worker whose prompt they are going back into.
    """
    if call.name == "FIND":
        return None
    if call.arg.startswith("/") or any(part == ".." for part in call.arg.split("/")):
        return (f"error: {call.arg} is outside the repository root; paths are relative to it and "
                "may not contain `..`")
    return None


def _payload(log: str) -> tuple[str, bool]:
    """The fenced observation, and whether the reader stopped early. Raises if the fence is absent.

    THE FIRST OPENING MARKER AND THE LAST CLOSING ONE, so that a source file quoting either of them
    is carried through whole rather than cutting its own observation short; and the truncation flag
    is read from AFTER the closing marker, which is the only region of the log the payload cannot
    reach. Both are content-forgery refusals, and the second was measured (`TRUNCATED`).
    """
    from .benchmarks import sandbox

    begin, end = log.find(OBS_BEGIN), log.rfind(OBS_END)
    if begin < 0 or end < begin:
        raise sandbox.SandboxError(
            "a read produced no observation markers, so the container did not run the reader: "
            f"{log.strip()[-400:]}")
    return log[begin + len(OBS_BEGIN):end].strip("\n"), TRUNCATED in log[end:]


def _call_line(call: ToolCall) -> str:
    """The call as the transcript shows it — rendered from the VALIDATED `ToolCall`, not from the
    worker's raw line, so nothing that failed to parse can re-enter a prompt through the echo."""
    return f"{call.name} {call.arg}" + (f" {call.start}" if call.name == "READ" else "")
