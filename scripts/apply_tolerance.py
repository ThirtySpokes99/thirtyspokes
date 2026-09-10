"""What `git apply` actually tolerates — the premise the worker read channel rests on.

MEASURED 2026-09-01. THE ANSWER: `git apply` matches on CONTEXT LINES alone. Wrong line numbers
and fabricated blob hashes are both tolerated; one invented context line refuses the patch.

    baseline (a real `git diff` patch)                ACCEPTED
    fabricated blob hashes `1234567..abcdefg`         ACCEPTED
    hunk header off by 18 lines                       ACCEPTED
    hunk header off by 198 lines                      ACCEPTED
    fabricated hashes AND off by 198                  ACCEPTED
    ONE context line invented                         rejected

WHY THIS SCRIPT EXISTS. the corpus record §1.2 measured R2E-Gym under the one-shot
scaffold at a routable band of +0.0000, with **0 of 75 episodes producing a patch `git apply`
would accept** while the gold patch scored exactly 1.0 on 24 of 25 of those same tasks through the
identical grader. The diagnosis — that the workers' patches were syntactically valid diffs whose
blob hashes, line numbers and context lines were all invented, and that only the context mattered —
is what the read channel (`v3/tools.py`) was designed against: it supplies the bytes of the file
and nothing else.

That diagnosis was asserted in three documents and measured in none. It is the load-bearing claim
of the whole design, so it is measured here. If it were false — if `git apply` also required
correct line numbers — then reading a file would not be sufficient and the channel would be aimed
at the wrong failure.

A NOTE ON METHOD, because the first three attempts at this got it wrong in the same way. Do NOT
hand-write the patch. A hand-written unified diff with leading context but no trailing context is
rejected for reasons that have nothing to do with the perturbation under test, which reads as
"`git apply` is strict" and is a false negative. Generate a genuine patch with `git diff`, then
perturb exactly one field of it, which is what `_perturb` does.

Run: `python scripts/apply_tolerance.py` — needs `git`, no network, no Docker, no model.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import tempfile

CASES: tuple[tuple[str, str, str], ...] = (
    # label, regex over one line of the real patch, replacement
    ("baseline (a real `git diff` patch)", "", ""),
    ("fabricated blob hashes", r"^index .*$", "index 1234567..abcdefg 100644"),
    ("hunk header off by 18 lines", r"^@@ -2,7 \+2,7 @@.*$", "@@ -20,7 +20,7 @@"),
    ("hunk header off by 198 lines", r"^@@ -2,7 \+2,7 @@.*$", "@@ -200,7 +200,7 @@"),
    ("ONE context line invented", r"^     b = 2$", "     b = 99"),
)


def _git(repo: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(("git", *args), cwd=repo, capture_output=True, text=True)


def _fixture(repo: pathlib.Path) -> str:
    """A one-commit repo, and the genuine patch that turns `return a` into `return a + b`.

    The filler lines matter: without a file long enough to hold a 198-line offset, the
    off-by-198 case would be refused for running past EOF rather than for its line numbers.
    """
    body = "def f():\n    a = 1\n    b = 2\n    c = 3\n    return a\n"
    body += "".join(f"# filler {i}\n" for i in range(1, 41))
    (repo / "m.py").write_text(body, encoding="utf-8")
    _git(repo, "init", "-q", ".")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "m.py")
    _git(repo, "commit", "-qm", "init")
    (repo / "m.py").write_text(body.replace("    return a\n", "    return a + b\n"), encoding="utf-8")
    patch = _git(repo, "diff").stdout
    _git(repo, "checkout", "--", "m.py")
    return patch


def _perturb(patch: str, pattern: str, replacement: str) -> str:
    if not pattern:
        return patch
    out, n = re.subn(pattern, replacement, patch, count=1, flags=re.MULTILINE)
    if n != 1:
        raise AssertionError(f"perturbation {pattern!r} matched {n} lines, not 1 — "
                             "the fixture patch changed shape and the case is no longer testing "
                             "what its label says")
    return out


def main() -> None:
    print(__doc__)
    with tempfile.TemporaryDirectory() as tmp:
        repo = pathlib.Path(tmp)
        patch = _fixture(repo)
        results = {}
        for label, pattern, replacement in CASES:
            candidate = repo / "candidate.patch"
            candidate.write_text(_perturb(patch, pattern, replacement), encoding="utf-8")
            ok = _git(repo, "apply", "--check", "candidate.patch").returncode == 0
            results[label] = ok
            print(f"  {label:<44} {'ACCEPTED' if ok else 'rejected'}")

        # Both-at-once: the shape every measured R2E-Gym patch actually had.
        both = _perturb(_perturb(patch, CASES[1][1], CASES[1][2]), CASES[2][1], "@@ -200,7 +200,7 @@")
        (repo / "candidate.patch").write_text(both, encoding="utf-8")
        ok = _git(repo, "apply", "--check", "candidate.patch").returncode == 0
        results["fabricated hashes AND off by 198"] = ok
        print(f"  {'fabricated hashes AND off by 198':<44} {'ACCEPTED' if ok else 'rejected'}")

    # The conclusion is an assertion, not a hope: if a future git changes this, the script fails
    # rather than printing a table nobody re-reads.
    assert results["baseline (a real `git diff` patch)"], "the fixture itself does not apply"
    assert results["fabricated blob hashes"], "blob hashes now matter — the read channel premise moved"
    assert results["hunk header off by 198 lines"], "line numbers now matter — the premise moved"
    assert not results["ONE context line invented"], "context no longer matters — the premise moved"
    print("\n  => context lines are the whole check. Supplying the bytes of the file is therefore\n"
          "     necessary AND sufficient to remove the measured failure mode, which is what\n"
          "     `v3/tools.py` READ exists to do.")


if __name__ == "__main__":
    main()
