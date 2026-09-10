"""The worker read channel: the grammar, the caps, and the confinement (§2.1, `v3/tools.py`).

THE SECURITY STORY OF THIS MODULE IS ROOT CONFINEMENT, AND IT IS TESTED ADVERSARIALLY RATHER THAN
REVIEWED. A miner can route to a model they control, so every tool the worker can call is a tool a
miner can drive — and there is a specific prize: `/r2e_tests` sits OUTSIDE `/testbed`, which
`r2egym.py` says in as many words is what makes the benchmark safe to grade. A read that escapes the
root hands a miner the graded test file, and a patch that special-cases it scores 1.0 without fixing
anything. So the escape is attempted here five ways — `..`, an absolute path, a symlinked directory,
a symlinked file, and `FIND` walking into a symlink — against a real fake root on the filesystem.

NOTHING HERE STARTS A CONTAINER. `benchmarks.mock.in_process` runs the SHIPPED `tools.READER` source
in this interpreter, so the confinement under test is the confinement that ships rather than a second
implementation of it, and the whole suite stays offline (no Docker, no network, no image, no money).
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from thirtyspokes.v3 import tools
from thirtyspokes.v3.benchmarks.mock import in_process
from thirtyspokes.v3.benchmarks.sandbox import SandboxError
from thirtyspokes.v3.config import (
    FIND_MAX,
    LIST_MAX,
    MAX_READS_PER_DELEGATE,
    OBSERVATION_BYTES,
    READ_LINES,
    TOOL_TIMEOUT_SECONDS,
)
from thirtyspokes.v3.types import Environment, Observation, ToolCall

# What a miner would be reaching for. `r2egym.py` puts the graded tests at `/r2e_tests`, outside the
# repository root — so in the fixture below it is a sibling of the root, reachable only by escaping.
GRADED_TESTS = "the graded test file"


@pytest.fixture
def world(tmp_path: pathlib.Path) -> Environment:
    """A repository root with a sibling directory holding the graded tests, and two ways out of it.

    Both symlinks are the real attack: `git apply` refuses to write through one (which is why the
    grader is safe), but nothing about a READ refuses one except the reader's own `realpath` check.
    """
    root, outside = tmp_path / "testbed", tmp_path / "r2e_tests"
    (root / "pkg").mkdir(parents=True)
    outside.mkdir()
    (outside / "test_gold.py").write_text(GRADED_TESTS)
    (root / "pkg" / "mod.py").write_text("import os\n\n\ndef broken(chunk):\n    return len(chunk)\n")
    (root / "README").write_text("a repository\n")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    (root / "pkg" / "leak.py").symlink_to(outside / "test_gold.py")
    return Environment(image="fixture:mock", root=str(root))


def observe(world: Environment, call: ToolCall) -> Observation:
    return tools.observe(world, call, "sha-of-the-reply", execute=in_process)


# --- the grammar (§3.2) ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("READ pkg/mod.py", ToolCall("READ", "pkg/mod.py")),
    ("READ pkg/mod.py 40", ToolCall("READ", "pkg/mod.py", 40)),
    ("LIST .", ToolCall("LIST", ".")),
    ("FIND def broken", ToolCall("FIND", "def broken")),          # a pattern may hold spaces
    ("Here is what I need.\n\nREAD pkg/mod.py\n\n", ToolCall("READ", "pkg/mod.py")),
])
def test_the_last_non_blank_line_is_the_call(raw: str, expected: ToolCall):
    """THE LAST NON-BLANK LINE DECIDES, which is inverted relative to `action.parse` where the last
    line MUST be an action. Deliberate: a Conductor that cannot emit an action is not routing,
    whereas a worker that ignores this channel must not be punished for ignoring it."""
    assert tools.parse_tool(raw) == expected


@pytest.mark.parametrize("raw", [
    "",
    "   \n\n  ",
    "READ",                              # no argument
    "read pkg/mod.py",                   # case: refused, never folded, like a model id
    "LIST",
    "FIND",
    "READ pkg/mod.py 0",                 # lines are numbered from 1
    "READ pkg/mod.py -3",
    "READ pkg/mod.py ٣",            # an Arabic-Indic digit is not a decimal
    "READ\tpkg/mod.py",                  # the grammar is space-separated
    "OPEN pkg/mod.py",
    "READ pkg/mod.py\nhere is my patch",  # the call is not the last line
    "diff --git a/x b/x\n-old\n+new",
])
def test_anything_that_is_not_exactly_one_call_is_the_submission(raw: str):
    """`None` means "this reply IS the answer" — never a repaired call, which would be the scaffold
    guessing at what a worker meant and then charging it for the guess."""
    assert tools.parse_tool(raw) is None


def test_an_oversized_argument_is_not_a_call_so_a_memorised_patch_cannot_ride_in_it():
    """The laundering shape worth naming: `FIND <2 KB of memorised gold patch>` renders the pattern
    into the transcript, so an unbounded argument would be a way to get a cheap model's bytes into a
    prompt. Over `MAX_ARG_BYTES` it is not a tool call at all — the reply is the submission and is
    graded, which is exactly what it would have been without the channel."""
    assert tools.parse_tool("FIND " + "x" * (tools.MAX_ARG_BYTES + 1)) is None
    assert tools.parse_tool("FIND " + "x" * tools.MAX_ARG_BYTES) is not None


def test_a_tool_call_cannot_hold_more_than_one_line_or_a_nul():
    """`types.ToolCall` refuses at construction what `parse_tool` refuses at the seam, so the type
    itself cannot carry a multi-line payload into a transcript a later prompt is composed from —
    the same guard `Action` gives on the Conductor's side. `str.splitlines` breaks on eleven
    characters, which is wider than `\\n` and the safe direction (`tests/test_invariant.py`)."""
    for bad in ("", "a\nb", "a\u2028b", "a\x00b"):
        with pytest.raises(ValueError, match="one non-empty line"):
            ToolCall("READ", bad)
    with pytest.raises(ValueError, match="not one of"):
        ToolCall("BASH", "rm -rf /")


def test_the_known_misfire_is_bounded_and_never_a_wrong_grade():
    """RECORDED RATHER THAN HIDDEN: a diff whose last context line reads `READ something` is
    consumed as a call. The cost is bounded — the loop re-asks, and the final turn always submits
    whatever comes back — so the worst case is wasted turns on a patch that is then still graded."""
    assert tools.parse_tool("@@ -1 +1 @@\n-x\n+y\n READ setup.py") == ToolCall("READ", "setup.py")
    # It is narrow, because a path holds no spaces: the same line with a second word is a diff again.
    assert tools.parse_tool("@@ -1 +1 @@\n-x\n+y\n READ the docs") is None


# --- composition: the §2.1 audit sentence ---------------------------------------------------------
def test_composing_with_no_observations_is_the_prompt_byte_for_byte():
    """The compatibility property the whole design rests on. `compose(p, ()) == p` is what makes the
    audit `request.text == compose(task.prompt, request.observations)` IDENTICAL to today's
    `request.text == task.prompt` on every benchmark that declares no tools and on the first turn of
    every delegate — so nothing already measured is re-scored."""
    for prompt in ("", "x", "Repository: sympy\n\n# Problem\n...", tools.PROTOCOL):
        assert tools.compose(prompt, ()) == prompt


def test_composition_is_a_pure_function_of_the_prompt_and_the_transcript(world: Environment):
    """It is PINNED and PURE, which is what makes the audit a recomputation rather than a claim
    about source code: anyone holding the published benchmark and the published record can rebuild
    the exact bytes a worker was sent."""
    first = observe(world, ToolCall("LIST", "."))
    second = observe(world, ToolCall("READ", "pkg/mod.py"))

    once = tools.compose("PROMPT", (first, second))
    assert once == tools.compose("PROMPT", (first, second))
    assert once.startswith("PROMPT")
    assert "LIST ." in once and "READ pkg/mod.py 1" in once
    assert "def broken" in once
    assert f"{MAX_READS_PER_DELEGATE - 2} more time(s)" in once, (
        "the remaining-read count is derived from the transcript; without it a worker spends its "
        "last turn on a call that is then graded as a patch")


def test_the_transcript_echoes_the_validated_call_and_not_the_workers_raw_line():
    """The echo is rendered from the `ToolCall`, so nothing that failed to parse can re-enter a
    prompt through it — and the line is single by construction (`ToolCall.__post_init__`)."""
    call = ToolCall("FIND", "def broken")
    composed = tools.compose("P", (Observation(call, "sha", "pkg/mod.py:4:def broken(chunk):", False),))
    assert "$ FIND def broken\n" in composed


# --- the three verbs, against a real root ---------------------------------------------------------
def test_read_returns_numbered_lines_from_the_requested_start(world: Environment):
    """The measured missing input, supplied: `git apply` matches CONTEXT LINES, and a worker writing
    a diff needs the line NUMBERS as well as the text (the sandbox record §7.2)."""
    whole = observe(world, ToolCall("READ", "pkg/mod.py"))
    tail = observe(world, ToolCall("READ", "pkg/mod.py", 4))

    assert whole.output.splitlines()[0] == "     1\timport os"
    assert tail.output.splitlines()[0] == "     4\tdef broken(chunk):"
    assert not whole.truncated and not tail.truncated
    assert whole.response_sha256 == "sha-of-the-reply"


def test_list_and_find_are_deterministic_and_ordered(world: Environment):
    """Sorted listings, hits ordered by (path, line), names only — no sizes, no timestamps, no inode
    data. That is what keeps an observation re-executable against the pinned image by a third party,
    and it is the property that dies the instant a write tool or a test runner is admitted."""
    listing = observe(world, ToolCall("LIST", "."))
    found = observe(world, ToolCall("FIND", "def broken"))

    assert listing.output.splitlines() == ["README", "escape", "pkg"]
    assert found.output == "pkg/mod.py:4:def broken(chunk):"
    assert observe(world, ToolCall("LIST", ".")) == listing
    assert observe(world, ToolCall("FIND", "def broken")) == found


def test_a_pattern_is_a_fixed_string_and_never_a_regex(world: Environment):
    """An attacker-chosen regex is catastrophic backtracking on the sandbox host for no gain over a
    fixed-string search, so the pattern is matched with `in` and a regex simply finds nothing."""
    assert observe(world, ToolCall("FIND", "def.*broken")).output == ""
    assert observe(world, ToolCall("FIND", "(a+)+$")).output == ""


# --- confinement: the whole security story (§6.2) -------------------------------------------------
@pytest.mark.parametrize("call", [
    ToolCall("READ", "../r2e_tests/test_gold.py"),        # layer 2: refused before a container
    ToolCall("READ", "pkg/../../r2e_tests/test_gold.py"),
    ToolCall("LIST", "/r2e_tests"),                       # layer 2: absolute
    ToolCall("READ", "/etc/passwd"),
    ToolCall("READ", "escape/test_gold.py"),              # layer 3: through a symlinked directory
    ToolCall("LIST", "escape"),
    ToolCall("READ", "pkg/leak.py"),                      # layer 3: through a symlinked FILE
])
def test_no_read_can_reach_the_graded_tests_outside_the_root(world: Environment, call: ToolCall):
    """THE ATTACK THAT MATTERS. `/r2e_tests` is outside `/testbed` and that is exactly what makes
    R2E-Gym safe to grade; a read that escapes hands a miner the file its patch will be graded
    against, and a patch that special-cases it scores 1.0 without fixing anything.

    A refusal is an OBSERVATION, not an exception: the worker got it wrong, which is data about the
    worker and costs it a turn (§8b.3). It is the container that failing raises."""
    observed = observe(world, call)

    assert GRADED_TESTS not in observed.output
    assert "outside the repository root" in observed.output
    assert not observed.truncated


def test_find_does_not_walk_out_through_a_symlink(world: Environment):
    """The third layer's other half: `os.walk(followlinks=False)` does not descend the symlinked
    directory, and every file it does reach has its realpath re-checked before it is opened — so the
    symlinked FILE inside the tree is skipped too. Two escapes, one assertion."""
    found = observe(world, ToolCall("FIND", GRADED_TESTS))

    assert found.output == ""
    assert not found.truncated, "an empty result must not read as a capped one"


@pytest.mark.parametrize("arg", ["../r2e_tests/test_gold.py", "/r2e_tests/test_gold.py",
                                 "pkg/../../r2e_tests/test_gold.py"])
def test_a_path_refused_before_the_container_starts_nothing(world: Environment, arg: str):
    """Layer 2 is not merely redundant with layer 3: it is what makes an escape attempt free. A
    container per hostile read is 0.177 s of an arm's two-hour clock, and a worker a miner controls
    can emit one every turn.

    BOTH of layer 2's clauses are exercised. Dropping the absolute-path half is a mutant that
    SURVIVED the suite as first written — layer 3 still refused it, so nothing failed, and the only
    lost property was the free refusal this test exists for."""
    def refuse(env, request):
        pytest.fail("a container was started for a path the parser could already refuse")

    observed = tools.observe(world, ToolCall("READ", arg), "sha", execute=refuse)
    assert "outside the repository root" in observed.output


def test_a_sibling_of_the_root_whose_name_merely_starts_with_it_is_still_outside(
        tmp_path: pathlib.Path):
    """`/testbed_secrets` is not under `/testbed`, and a prefix comparison would say it is.

    The mutant that found this: `target.startswith(root)` instead of `startswith(root + os.sep)`.
    It survived every other confinement test, because reaching a same-prefix SIBLING needs a symlink
    aimed at one rather than a `..` — which is exactly what a benchmark image could contain."""
    root, sibling = tmp_path / "testbed", tmp_path / "testbed_secrets"
    root.mkdir()
    sibling.mkdir()
    (sibling / "gold.py").write_text(GRADED_TESTS)
    (root / "src").symlink_to(sibling, target_is_directory=True)
    env = Environment(image="fixture:mock", root=str(root))

    for call in (ToolCall("READ", "src/gold.py"), ToolCall("LIST", "src"),
                 ToolCall("FIND", GRADED_TESTS)):
        assert GRADED_TESTS not in observe(env, call).output


# --- caps, truncation and the two failure classes (§6.3, §8b.3) -----------------------------------
def test_every_cap_is_enforced_and_truncation_is_marked_never_silent(tmp_path: pathlib.Path):
    """A replay must be able to tell a whole file from the first `READ_LINES` of one, so the reader
    marks that it stopped and `Observation.truncated` carries it into the record. Silence would be a
    lie about what the worker was shown."""
    root = tmp_path / "testbed"
    (root / "many").mkdir(parents=True)
    (root / "long.py").write_text("".join(f"line {i}\n" for i in range(READ_LINES + 50)))
    for i in range(LIST_MAX + 10):
        (root / "many" / f"f{i:04d}.py").write_text("hit\n")
    env = Environment(image="fixture:mock", root=str(root))

    read = observe(env, ToolCall("READ", "long.py"))
    listing = observe(env, ToolCall("LIST", "many"))
    found = observe(env, ToolCall("FIND", "hit"))

    assert len(read.output.splitlines()) == READ_LINES and read.truncated
    assert len(listing.output.splitlines()) == LIST_MAX and listing.truncated
    assert len(found.output.splitlines()) == FIND_MAX and found.truncated
    assert all(len(o.output) <= OBSERVATION_BYTES for o in (read, listing, found))
    assert tools.TRUNCATED in tools.compose("P", (read,))


def test_one_observation_cannot_outgrow_the_prompt_budget(tmp_path: pathlib.Path):
    """`OBSERVATION_BYTES` is what bounds the loop's DOLLAR cost — each read turn adds at most this
    much to the next turn's input — so a single enormous file must not be able to breach it."""
    root = tmp_path / "testbed"
    root.mkdir()
    (root / "big.py").write_text("x" * (OBSERVATION_BYTES * 4) + "\n")
    env = Environment(image="fixture:mock", root=str(root))

    observed = observe(env, ToolCall("READ", "big.py"))

    assert len(observed.output) <= OBSERVATION_BYTES and observed.truncated


def test_a_container_that_ignores_the_budget_is_capped_and_marked_by_observe_anyway():
    """`observe`'s own cap is NOT redundant with the reader's, and this is the difference: the reader
    is our source, but the INTERPRETER running it belongs to the benchmark's image. A container that
    returned more than the pin would otherwise put an unbounded payload into the next turn's prompt
    — real dollars, silently — and report it as a complete read. Both mutants (drop the cap, drop
    the flag it implies) survived a suite that only ever exercised the reader's own budget."""
    def over_budget(env, request):
        return f"{tools.OBS_BEGIN}\n{'y' * (OBSERVATION_BYTES * 3)}\n{tools.OBS_END}\n"

    observed = tools.observe(Environment("x", "/testbed"), ToolCall("READ", "a.py"), "sha",
                             execute=over_budget)

    assert len(observed.output) == OBSERVATION_BYTES
    assert observed.truncated, "a payload the pin had to cut is a truncated one, whoever cut it"


@pytest.mark.parametrize("call", [ToolCall("READ", "nope.py"), ToolCall("LIST", "pkg/mod.py"),
                                  ToolCall("READ", "pkg/mod.py", 900)])
def test_a_call_the_worker_got_wrong_is_feedback_and_not_an_exclusion(world: Environment,
                                                                      call: ToolCall):
    """§6.3's first class. A path that does not exist, a file asked to be a directory, a start past
    EOF: all data about the WORKER's behaviour, so each comes back as an observation carrying an
    error string. It costs a turn, it is deterministic, and it is something the worker can act on."""
    observed = observe(world, call)

    assert observed.output.startswith("error:")
    assert not observed.truncated


def test_a_container_that_produced_no_markers_is_a_sandbox_failure_and_not_an_empty_observation():
    """§6.3's second class, and the rule `r2egym.r2e_parse_log` already applies to a missing pytest
    summary block: absence of the fence means the reader did not run, which is a fact about the
    VALIDATOR. It raises, so the caller drops the task from BOTH arms and from the denominator
    (§6.3c) — an empty observation would charge a miner for the owner's Docker."""
    def broken(env, request):
        return "Killed\n"

    with pytest.raises(SandboxError, match="no observation markers"):
        tools.observe(Environment("x", "/testbed"), ToolCall("READ", "a.py"), "sha", execute=broken)


def test_the_reader_is_handed_every_cap_rather_than_baking_them_in(world: Environment):
    """The pins stay in `config.py`: the reader is a pure function of the JSON it is given, so a
    replay can read what it was bounded by instead of inferring it from a vendored copy."""
    seen: list[dict] = []

    def record(env, request):
        seen.append(dict(request))
        return f"{tools.OBS_BEGIN}\n\n{tools.OBS_END}\n"

    tools.observe(world, ToolCall("READ", "pkg/mod.py", 7), "sha", execute=record)

    assert seen == [{"name": "READ", "arg": "pkg/mod.py", "start": 7, "root": world.root,
                     "read_lines": READ_LINES, "list_max": LIST_MAX, "find_max": FIND_MAX,
                     "max_bytes": OBSERVATION_BYTES, "hidden": list(tools.HIDDEN)}]


# --- what the SOURCE can forge, and the prize that sits inside the root ----------------------------
def test_a_file_that_quotes_the_markers_comes_back_whole_and_unflagged(tmp_path: pathlib.Path):
    """THE OBSERVATION IS FENCED, AND ITS CONTENT MUST NOT BE ABLE TO MOVE THE FENCE.

    Three forgeries, each measured against the code as first written, all now refused:

      * the closing marker in a source line cut the observation short there, because `_payload`
        took the FIRST closing marker (mutant M24: `rfind` -> `find` survived the suite);
      * the OPENING marker in a source line did the same from the other end, and this half was
        covered by nothing: mutating `log.find(OBS_BEGIN)` to `rfind` survived all 1420 tests.
        `_payload` promises "the first opening marker and the LAST closing one" and only the second
        clause was measured — under the mutation, `one\\n<BEGIN>\\nthree` showed the worker `three`
        alone, i.e. a source deciding how much of itself the worker is shown;
      * a source line reading exactly `[truncated]` was DELETED from the observation and the worker
        was told the read had been cut — `x = 1\\n[truncated]\\n` came back as `1\\tx = 1` with
        `truncated=True`. The flag is now emitted AFTER the closing marker, i.e. in the one region
        of the container log the payload cannot reach; in-band it was a signal the source owned."""
    root = tmp_path / "testbed"
    root.mkdir()
    (root / "quoting.py").write_text(f"a = 1\nb = '{tools.OBS_END}'\nc = 3\n")
    (root / "opening.py").write_text(f"a = 1\nb = '{tools.OBS_BEGIN}'\nc = 3\n")
    (root / "forging.py").write_text(f"x = 1\n{tools.TRUNCATED}\n")
    env = Environment(image="fixture:mock", root=str(root))

    quoting = observe(env, ToolCall("READ", "quoting.py"))
    opening = observe(env, ToolCall("READ", "opening.py"))
    forging = observe(env, ToolCall("READ", "forging.py"))

    assert len(quoting.output.splitlines()) == 3 and "c = 3" in quoting.output
    assert not quoting.truncated
    assert len(opening.output.splitlines()) == 3 and "a = 1" in opening.output
    assert not opening.truncated
    assert forging.output.splitlines()[-1].endswith(tools.TRUNCATED)
    assert not forging.truncated, "file content must not be able to forge the truncation flag"


def test_the_repositorys_own_history_is_not_readable(tmp_path: pathlib.Path):
    """`.git` IS INSIDE THE ROOT, AND IT IS THE ONE PRIZE ROOT CONFINEMENT CANNOT REACH.

    `r2egym.py` says twice that `/testbed` is the git repository, so the read channel's boundary
    contains the repository's whole history — and a clone that kept its upstream refs holds the very
    commit the task asks the worker to re-derive. The reader cannot verify what a third party's image
    contains, so metadata is excluded rather than trusted (`tools.HIDDEN`), through the realpath so
    that a symlink named like source does not launder it."""
    root = tmp_path / "testbed"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "ORIG_HEAD").write_text("the fix commit\n")
    (root / "src").symlink_to(root / ".git", target_is_directory=True)
    # A FILE symlink into `.git`, which is NOT the same case as the directory symlink above and was
    # the one that leaked. `os.walk(followlinks=False)` refuses to DESCEND a linked directory, so
    # the directory case never reached `find`'s loop and passed for a reason that did not
    # generalise; a linked FILE is listed by the real directory it sits in and does reach it.
    # Measured before the fix: `READ notes.py` raised "repository metadata, not source" while
    # `FIND "the fix commit"` returned `notes.py:1:the fix commit`. `find` was testing
    # `inside(...) is None` and discarding the realpath instead of running `metadata` on it — one
    # rule, two implementations, and this was the implementation missing it.
    (root / "notes.py").symlink_to(root / ".git" / "ORIG_HEAD")
    (root / "mod.py").write_text("x = 1\n")
    env = Environment(image="fixture:mock", root=str(root))

    for call in (ToolCall("READ", ".git/ORIG_HEAD"), ToolCall("LIST", ".git"),
                 ToolCall("READ", "src/ORIG_HEAD"), ToolCall("LIST", "src"),
                 ToolCall("READ", "notes.py"), ToolCall("FIND", "the fix commit")):
        assert "the fix commit" not in observe(env, call).output, (
            f"{call.name} {call.arg} reached the repository's own history")
    assert "x = 1" in observe(env, ToolCall("READ", "mod.py")).output, "source is still readable"


def test_a_hidden_entry_is_omitted_from_a_listing_and_not_merely_refused_on_use(
        tmp_path: pathlib.Path):
    """HIDDEN is applied to `LIST`'s output, not only to the paths the other verbs accept.

    Every use of `.git` is already refused — `READ .git/...`, `LIST .git`, and `FIND` prunes the
    walk — so a name shown in the root listing can never be acted on. It can only cost the worker
    one of its `MAX_READS_PER_DELEGATE` reads to discover that, which is a real price: the census
    beside that constant puts three reads at 92.2% of the corpus served and two at 71.1%. Hiding the
    name removes a trap without removing any capability.

    THE SECOND HALF IS WHAT KEEPS THIS HONEST: `HIDDEN` is a set of names, NOT a dotfile rule.
    `.gitignore` is a dotfile the worker may legitimately want and it stays listed and readable, so
    a future `hidden` that quietly became `startswith(".")` fails here.
    """
    root = tmp_path / "testbed"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / ".gitignore").write_text("*.pyc\n")
    (root / "src").mkdir()
    env = Environment(image="fixture:mock", root=str(root))

    listed = observe(env, ToolCall("LIST", ".")).output
    assert ".git\n" not in listed and not listed.startswith(".git\n"), (
        "a hidden entry was advertised in the listing that every other verb refuses")
    assert ".gitignore" in listed and "src" in listed, (
        "HIDDEN is a set of names; it must not have become a blanket dotfile filter")
    assert "*.pyc" in observe(env, ToolCall("READ", ".gitignore")).output


def test_find_is_ordered_by_path_over_a_tree_the_filesystem_returns_unordered(
        tmp_path: pathlib.Path):
    """DETERMINISM IS THE PROPERTY THE WHOLE AUDIT RESTS ON: a replay must reach the same verdict, and
    hit ORDER is observation bytes, which is prompt bytes, which is the answer.

    `os.walk` yields directory entries in filesystem order, which is not sorted on any filesystem
    these images use. Dropping the two `sort()` calls was a mutant that SURVIVED, because the earlier
    ordering test used a tree small enough that the two orders coincided."""
    root = tmp_path / "testbed"
    root.mkdir()
    directories = ("zeta", "alpha", "mid", "beta", "gamma", "delta", "eps", "omega")
    for name in directories:
        (root / name).mkdir()
        # BOTH sorts matter and they fail separately: `dirs` decides which directory comes first and
        # `files` decides which file inside one does, so each needs a tree wide enough to tell.
        for leaf in ("z", "a", "m", "b", "g"):
            (root / name / f"{leaf}.py").write_text("needle\n")
    env = Environment(image="fixture:mock", root=str(root))

    hits = observe(env, ToolCall("FIND", "needle")).output.splitlines()

    assert hits == sorted(hits)
    assert [h.split("/")[0] for h in hits[:5]] == [sorted(directories)[0]] * 5
    assert [h.split("/")[1].split(":")[0] for h in hits[:5]] == ["a.py", "b.py", "g.py", "m.py",
                                                                 "z.py"]


def test_the_readers_own_output_is_bounded_so_the_markers_can_never_be_cut_off(
        tmp_path: pathlib.Path):
    """The container-side byte budget is not redundant with `observe`'s cap on the OBSERVATION.

    What it bounds is the LOG, and the log is what the fence lives in: a reader that emitted
    megabytes would be at the mercy of whatever truncates a container's output, and a payload whose
    markers were cut is a `SandboxError` — one task out of both arms for a file that was merely
    large. Ignoring the budget was a mutant that survived, because `observe`'s outer cap hid it."""
    root = tmp_path / "testbed"
    root.mkdir()
    (root / "huge.py").write_text("".join(f"{'q' * 300}{i}\n" for i in range(READ_LINES)))
    env = Environment(image="fixture:mock", root=str(root))

    log = in_process(Environment(image="x", root=str(root)),
                     {"name": "READ", "arg": "huge.py", "start": 1, "root": str(root),
                      "read_lines": READ_LINES, "list_max": LIST_MAX, "find_max": FIND_MAX,
                      "max_bytes": OBSERVATION_BYTES, "hidden": list(tools.HIDDEN)})

    assert tools.OBS_BEGIN in log and tools.OBS_END in log
    # One line may overshoot the remaining budget, by at most the budget itself (`keep`).
    assert len(log) < 2 * OBSERVATION_BYTES + len(tools.READER)
    assert observe(env, ToolCall("READ", "huge.py")).truncated


def test_source_decodes_the_same_way_whatever_locale_the_images_interpreter_has(
        tmp_path: pathlib.Path):
    """`open(path, "r")` picks its encoding from the LOCALE, and the interpreter that runs the reader
    belongs to the BENCHMARK'S IMAGE, not to us — 13 repositories' worth of third-party images, each
    with whatever `LANG` its builder happened to set. Unpinned, `café` in a source file is one
    character under `C.UTF-8` and two replacement characters under `C`: different observation bytes,
    a different worker prompt, a different answer, and two validators replaying one reveal disagree.

    Measured in a child process under `LC_ALL=C` with PEP 538's coercion and PEP 540's UTF-8 mode
    both disabled, which is the only way to reach a non-UTF-8 default from a test — the same reason
    `test_benchmarks.py` spawns one for `PYTHONHASHSEED`."""
    root = tmp_path / "testbed"
    root.mkdir()
    (root / "accents.py").write_text("café = 1\n", encoding="utf-8")
    out = tmp_path / "payload"
    child = tmp_path / "child.py"
    child.write_text(
        "import json, pathlib, sys\n"
        "from thirtyspokes.v3.benchmarks.mock import in_process\n"
        "from thirtyspokes.v3.types import Environment\n"
        f"log = in_process(Environment('x', {str(root)!r}), json.loads(sys.argv[1]))\n"
        f"pathlib.Path({str(out)!r}).write_text(log, encoding='utf-8')\n")
    request = {"name": "READ", "arg": "accents.py", "start": 1, "root": str(root),
               "read_lines": READ_LINES, "list_max": LIST_MAX, "find_max": FIND_MAX,
               "max_bytes": OBSERVATION_BYTES, "hidden": list(tools.HIDDEN)}

    subprocess.run([sys.executable, str(child), json.dumps(request)], check=True,
                   env={**os.environ, "LC_ALL": "C", "PYTHONUTF8": "0",
                        "PYTHONCOERCECLOCALE": "0"})

    assert "café = 1" in out.read_text(encoding="utf-8")
    assert "café = 1" in observe(Environment("fixture:mock", str(root)),
                                 ToolCall("READ", "accents.py")).output


def test_a_read_goes_through_the_declared_sandbox_seam_with_no_shell_and_a_bounded_clock(
        monkeypatch: pytest.MonkeyPatch):
    """THE SEAM IS `sandbox.run` AND THERE IS NO SECOND ONE (§8b.9).

    Nothing here starts a container: `sandbox.run` is replaced and its arguments are asserted. What
    is being pinned is that `argv` is the interpreter plus two FILE paths — so the worker's argument
    is never a token on a command line — that the clock is `TOOL_TIMEOUT_SECONDS`, and that every
    host-protection cap is left at `sandbox.run`'s own default rather than relaxed at this call
    site. An unbounded timeout was a mutant that survived the suite as first written."""
    from thirtyspokes.v3.benchmarks import sandbox as seam

    seen: dict[str, object] = {}

    def fake_run(image, argv, *, mount, log_path, timeout, **caps):
        seen.update(image=image, argv=list(argv), timeout=timeout, caps=caps,
                    payload=sorted(p.name for p in mount.iterdir()),
                    reader=(mount / "reader.py").read_text(),
                    call=(mount / "call.json").read_text())
        log_path.write_text(f"{tools.OBS_BEGIN}\nfine\n{tools.OBS_END}\n")
        return seam.SandboxResult(exit_code=0, output="", log_path=log_path)

    monkeypatch.setattr(seam, "run", fake_run)
    monkeypatch.setenv(seam.DOCKER_HOST_ENV, seam.DOCKER_HOST_LOCAL)

    observed = tools.observe(Environment(image="r2e:abc", root="/testbed"),
                             ToolCall("READ", "pkg/mod.py", 7), "sha")

    assert observed.output == "fine"
    assert seen["image"] == "r2e:abc"
    # `-I` (isolated mode) is pinned here because it is a security flag, not a style choice.
    # MEASURED: without it, an image that sets PYTHONPATH substitutes the `json` module that runs
    # INSIDE the reader — the owner code enforcing realpath containment and HIDDEN — and these
    # images are third party and, per `r2egym.py`, under no pin. It implies `-E` and `-s`, and it
    # costs 0 ms. Dropping it must fail here rather than quietly hand the image that hook.
    assert seen["argv"] == ["python3", "-I", f"{seam.MOUNT}/reader.py", f"{seam.MOUNT}/call.json"]
    assert seen["timeout"] == TOOL_TIMEOUT_SECONDS
    assert seen["caps"] == {}, "every host-protection cap must stay at sandbox.run's own default"
    assert seen["payload"] == ["call.json", "reader.py"]
    assert seen["reader"] == tools.READER
    assert "pkg/mod.py" in str(seen["call"])


def test_in_sandbox_refuses_before_anything_runs_when_no_grading_host_is_declared(
        monkeypatch: pytest.MonkeyPatch):
    """§8b.9's declaration gate, reached through the READ path rather than the grading one: a read
    tool on the box that holds the chain hotkey is a hotkey-reading tool, and `V3_DOCKER_HOST` is
    what makes the split expressible. `run` refuses before a command line is built."""
    from thirtyspokes.v3.benchmarks import sandbox as seam

    monkeypatch.delenv(seam.DOCKER_HOST_ENV, raising=False)

    with pytest.raises(SandboxError, match=seam.DOCKER_HOST_ENV):
        tools.observe(Environment(image="r2e:abc", root="/testbed"), ToolCall("READ", "m.py"),
                      "sha")
