"""Challenger trees fetched, and kings published, ON THE SERVING HOST (§8b.9, §11-8).

WHY THE BYTES MOVE THERE. `<state>/trees` on the controller is a network mount of the serving host's
trees directory: vLLM loads a tree from the host's own disk, so that is where a tree has to end up.
Pulling a ~70 GB tree to that mount THROUGH the controller means every byte crosses the WAN into the
mount, and then crosses it again when the controller reads the file back to hash it — measured on
the owner's link, hours per challenger. The same objects pulled on the host itself from presigned
URLs, several at a time and hashed on its local disk, take minutes. Promotion had the mirror-image
cost: publishing a king read the whole tree back through the mount before sending it to R2.

WHAT DOES NOT MOVE. Every check that decides anything stays on the controller and runs first
(`store.fetch_submission`: the manifest, its digest against the commitment, the hotkey and
registration, the signature, the listing). The host is handed the committed size and digest of each
file, but it does not judge: it reports what it saw, one result line per file, and THIS module
compares every reported size and digest with the manifest itself before `store` writes the
`.verified` marker. After a fetch the controller also re-reads the tree through the mount the way
`admission.admit` will, so a `--serve-trees` that does not name the directory behind `<state>/trees`
is caught here rather than surfacing as a confusing refusal.

THE CREDENTIAL BOUNDARY. No R2 credential leaves the controller. The host receives one presigned
URL per object (or per part, for promotion): scoped to that object, expiring after
`PRESIGNED_TRANSFER_SECONDS`, written only into the script that travels on ssh's STDIN — never on
any command line, neither ssh's nor curl's (curl reads it from a `/dev/fd` config written by the
`printf` builtin), never on curl's stderr (discarded), and never in a log line or exception text
here. A presigned URL is still a bearer capability for its object until it expires; temporary
credentials cap that lifetime at their own expiry.

TWO KINDS OF FAILURE, AND THE DIFFERENCE IS A MINER'S SHOT. A plain `store.StoreError` is raised
only for evidence about the miner's tree: a file the host downloaded twice, at the committed size,
and hashed to a digest the signed manifest does not give. Everything else — a runner that failed
or timed out, output that is missing, garbled or undecodable, a curl failure, a mount that does not
show what the host wrote — is the owner's side and surfaces as `HostTransferError` (a
`serve.ServeError`), `subprocess.TimeoutExpired` or `OSError`, never as a plain `StoreError`. A
size the host observed that differs from the size the bucket LISTS is a transfer fault too: the
listing was already checked against the manifest on the controller before the host was asked.

ON THE HOST only bash (≥ 4.3, for `wait -n`), curl and coreutils are assumed. Fetch writes are
confined to `<serve-trees>/<registration id>/`; promotion only reads the tree.
"""

from __future__ import annotations

import re
import secrets
import shlex
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .serve import Runner, ServeError, ssh_runner
from .store import (MANIFEST_NAME, PART_SIZE, PARTIAL_MARKER, Manifest, ManifestFile, S3Bucket,
                    StoreError)

# The transfer profile on the host. NOT mechanism constants (nothing here reaches a score, a window
# or the schedule root), so they live beside the code that uses them rather than in `config.py`.
#
# How long one fetch or one promotion may run before the controller gives up on it. The measured
# host-side pull of a full tree is minutes; four hours covers a link roughly fifty times slower. A
# timeout is the owner's failure and defers, so erring long costs a slow window, while erring short
# would turn a slow link into a permanent deferral. Deliberately NOT `ssh_runner`'s 180 s default,
# which is sized for launching a server.
TREE_TRANSFER_TIMEOUT_SECONDS = 4 * 3600
# A URL must outlive the runner that uses it, or an expiry would surface as a 403 in the last hour
# of a slow transfer. SigV4 caps a presigned URL at seven days.
PRESIGNED_TRANSFER_SECONDS = TREE_TRANSFER_TIMEOUT_SECONDS + 3600
# Files (fetch) or parts (promotion) in flight at once — the manual host-side pull ran this many.
HOST_TRANSFER_PARALLELISM = 8
# Downloads per file. At least two, because a digest mismatch is only reported once two downloads
# agree on the same wrong bytes: one bad read on a flaky path must not spend a shot.
FETCH_ATTEMPTS = 3
PART_ATTEMPTS = 5
RETRY_PAUSE_SECONDS = 5
# The mount check after a fetch: sshfs caches attributes and directory listings, so a file the host
# just renamed into place can be invisible here for a few seconds. The mount's `dcache_timeout` and
# `attr_timeout` must stay below this budget (docs/VALIDATOR.md §2).
MOUNT_SETTLE_ATTEMPTS = 4
MOUNT_SETTLE_SECONDS = 15.0
# S3's part-count ceiling; with `store.PART_SIZE` it bounds one object at 640 GB.
MAX_PARTS = 10_000

_RECORD = "THIRTYSPOKES-"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MD5 = re.compile(r"^[0-9a-f]{32}$")
_ETAG = re.compile(r'^"?([0-9a-fA-F]{32})"?$')
_TREES = re.compile(r"^/[A-Za-z0-9._/-]*$")
_DIR_NAME = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")
_URL = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*)://[\x21-\x7e]+$")


class HostTransferError(ServeError):
    """The host could not move or report the bytes. OWNER-SIDE: it says nothing about the tree.

    A `ServeError` so that whatever classifies transport and serving failures as the owner's treats
    this the same way; never a `StoreError`, which would refuse a submission for our own outage.
    """


# --- the scripts ---------------------------------------------------------------------------------
#
# Rendered by `_render`, which substitutes `@@NAME@@` tokens with values that are either
# `shlex.quote`d or validated digits. The file list travels in a heredoc whose delimiter is QUOTED
# (no expansion inside) and carries a random nonce; every data line holds tabs, so none can equal
# the delimiter. Every expansion in the scripts is double-quoted. A result line is one `printf`
# well under PIPE_BUF, so lines from parallel jobs cannot interleave.

_PREAMBLE = r"""set -u -o pipefail
export LC_ALL=C
# A proxy in the host's shell environment would sit between curl and R2, and a curlrc could add one
# (`-q` below ignores it): bytes shown to be wrong must be the bucket's, never a middlebox's.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy
base=@@BASE@@
parallel=@@PARALLEL@@
attempts=@@ATTEMPTS@@
pause=@@PAUSE@@
proto=@@PROTO@@
trap 'kill $(jobs -rp) 2>/dev/null' EXIT
trap 'exit 129' HUP
trap 'exit 143' TERM
trap 'exit 141' PIPE
# A job whose result line hit a closed pipe died of SIGPIPE: the controller is gone, so stop.
throttle() {
  while (( $(jobs -rp | wc -l) >= parallel )); do wait -n; (( $? == 141 )) && exit 141; done
}
drain() {
  while (( $(jobs -rp | wc -l) > 0 )); do wait -n; (( $? == 141 )) && exit 141; done
  wait
}
"""

_FETCH_SCRIPT = _PREAMBLE + r"""umask 022
partial=@@PARTIAL@@
if [[ -L $base ]]; then echo "the tree directory is a symlink" >&2; exit 65; fi
mkdir -p -- "$base" || exit 73
declare -a sizes=() shas=() urls=() paths=()
declare -A wanted=()
count=0
while IFS=$'\t' read -r size sha url path; do
  sizes[count]=$size; shas[count]=$sha; urls[count]=$url; paths[count]=$path
  wanted["k:$path"]=1
  count=$((count + 1))
done <<'@@DELIM@@'
@@DATA@@@@DELIM@@

# Everything under the tree that the manifest does not name goes: leftovers of a killed run (ours or
# an older downloader's), and every symlink, so no write below can be redirected out of the tree.
shopt -s globstar nullglob dotglob
for entry in "$base"/**; do
  if [[ -L $entry ]]; then
    rm -f -- "$entry"
  elif [[ -f $entry ]]; then
    rel=${entry#"$base"/}
    [[ -n ${wanted["k:$rel"]+x} ]] || rm -f -- "$entry"
  fi
done
shopt -u globstar nullglob dotglob

report() { printf 'THIRTYSPOKES-FETCH\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6"; }

fetch_one() {
  local i=$1
  local size=${sizes[$i]} sha=${shas[$i]} url=${urls[$i]} path=${paths[$i]}
  local target="$base/$path" tmp seen digest rc=0 attempt last=""
  if [[ -f $target && ! -L $target ]]; then
    seen=$(stat -c %s -- "$target") || seen=""
    if [[ $seen == "$size" ]]; then
      digest=$(sha256sum < "$target" | cut -d' ' -f1) || digest=""
      if [[ $digest == "$sha" ]]; then report "$i" kept "$seen" "$digest" 0 "$path"; return; fi
    fi
  fi
  if ! mkdir -p -- "$(dirname -- "$target")" 2>/dev/null; then
    report "$i" failed - - 73 "$path"; return
  fi
  tmp="$target$partial-$BASHPID"
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    (( attempt == 1 )) || sleep "$pause"
    rm -f -- "$tmp"
    curl -q -sS --fail --proto "$proto" --noproxy '*' --retry 3 --connect-timeout 30 \
      --speed-limit 1024 --speed-time 120 -o "$tmp" -K <(printf 'url = "%s"\n' "$url") 2>/dev/null
    rc=$?
    (( rc == 0 )) || continue
    [[ -e $tmp ]] || : > "$tmp"
    seen=$(stat -c %s -- "$tmp") || { rc=74; continue; }
    digest=$(sha256sum < "$tmp" | cut -d' ' -f1) || { rc=74; continue; }
    if [[ $seen == "$size" && $digest == "$sha" ]]; then
      # Same directory, so the rename is atomic: the final name never holds a partial file.
      if mv -f -T -- "$tmp" "$target"; then report "$i" fetched "$seen" "$digest" 0 "$path"; return; fi
      rc=74; continue
    fi
    rm -f -- "$tmp"
    if [[ $last == "$seen:$digest" ]]; then report "$i" mismatch "$seen" "$digest" 0 "$path"; return; fi
    last="$seen:$digest"; rc=0
  done
  rm -f -- "$tmp"
  if [[ -n $last && $rc == 0 ]]; then report "$i" unstable - - 0 "$path"; else report "$i" failed - - "$rc" "$path"; fi
}

for ((i = 0; i < count; i++)); do
  throttle
  fetch_one "$i" </dev/null &
done
drain
printf 'THIRTYSPOKES-FETCH-DONE\t%s\n' "$count"
exit 0
"""

_UPLOAD_SCRIPT = _PREAMBLE + r"""if [[ ! -d $base || -L $base ]]; then echo "no tree directory on the host" >&2; exit 66; fi
work=$(mktemp -d) || exit 70
trap 'kill $(jobs -rp) 2>/dev/null; rm -rf -- "$work"' EXIT
declare -a fsizes=() fshas=() fpaths=() pfile=() pnum=() poff=() plen=() purl=()
files=0
parts=0
while IFS=$'\t' read -r kind a b c d e; do
  case $kind in
    F) fsizes[files]=$a; fshas[files]=$b; fpaths[files]=$c; files=$((files + 1)) ;;
    P) pfile[parts]=$a; pnum[parts]=$b; poff[parts]=$c; plen[parts]=$d; purl[parts]=$e
       parts=$((parts + 1)) ;;
  esac
done <<'@@DELIM@@'
@@DATA@@@@DELIM@@

hash_one() {
  local i=$1 file="$base/${fpaths[$1]}" seen digest status
  if [[ ! -f $file || -L $file ]]; then
    printf 'THIRTYSPOKES-HASH\t%s\tmissing\t-\t-\t%s\n' "$i" "${fpaths[$i]}"; return
  fi
  if ! seen=$(stat -c %s -- "$file") || ! digest=$(sha256sum < "$file" | cut -d' ' -f1); then
    printf 'THIRTYSPOKES-HASH\t%s\tfailed\t-\t-\t%s\n' "$i" "${fpaths[$i]}"; return
  fi
  status=mismatch
  if [[ $seen == "${fsizes[$i]}" && $digest == "${fshas[$i]}" ]]; then status=ok; : > "$work/ok.$i"; fi
  printf 'THIRTYSPOKES-HASH\t%s\t%s\t%s\t%s\t%s\n' "$i" "$status" "$seen" "$digest" "${fpaths[$i]}"
}

# The ETag from a header dump, whatever the transport: HTTP/2 names headers in lowercase, HTTP/1.1
# ends lines in CRLF, and an interim response (100 Continue) writes a block of its own. Only the
# LAST block counts, so a status line resets what was seen.
etag_of() {
  local line name value etag=""
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line%$'\r'}
    if [[ $line == HTTP/* ]]; then etag=""; continue; fi
    [[ $line == *:* ]] || continue
    name=${line%%:*}
    if [[ ${name,,} == etag ]]; then
      value=${line#*:}
      value=${value#"${value%%[![:space:]]*}"}
      value=${value%"${value##*[![:space:]]}"}
      etag=$value
    fi
  done < "$1"
  printf '%s' "$etag"
}

report_part() { printf 'THIRTYSPOKES-PART\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6"; }

put_part() {
  local j=$1
  local i=${pfile[$j]} n=${pnum[$j]} off=${poff[$j]} len=${plen[$j]} url=${purl[$j]}
  local file="$base/${fpaths[$i]}" hdr="$work/headers.$j" md5 etag rc=0 attempt
  if [[ ! -e $work/ok.$i ]]; then report_part "$i" "$n" skipped - - 0; return; fi
  # The part's MD5, from its own read of the file: R2 answers a part PUT with the MD5 of the bytes
  # it received, so the controller can tie what landed to what was read before completing.
  md5=$(dd if="$file" iflag=skip_bytes,count_bytes skip="$off" count="$len" bs=4M status=none \
        | md5sum | cut -d' ' -f1) || { report_part "$i" "$n" failed - - 75; return; }
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    (( attempt == 1 )) || sleep "$pause"
    : > "$hdr"
    dd if="$file" iflag=skip_bytes,count_bytes skip="$off" count="$len" bs=4M status=none \
      | curl -q -sS --fail --proto "$proto" --noproxy '*' --connect-timeout 30 \
          --speed-limit 1024 --speed-time 120 -X PUT --data-binary @- -H 'Content-Type:' \
          -H 'Expect:' -D "$hdr" -o /dev/null -K <(printf 'url = "%s"\n' "$url") 2>/dev/null
    rc=$?
    if (( rc == 0 )); then
      etag=$(etag_of "$hdr")
      if [[ -n $etag ]]; then report_part "$i" "$n" ok "$md5" "$etag" 0; return; fi
      rc=76
    fi
  done
  report_part "$i" "$n" failed "$md5" - "$rc"
}

for ((i = 0; i < files; i++)); do
  throttle
  hash_one "$i" </dev/null &
done
drain
for ((j = 0; j < parts; j++)); do
  throttle
  put_part "$j" </dev/null &
done
drain
printf 'THIRTYSPOKES-UPLOAD-DONE\t%s\t%s\n' "$files" "$parts"
exit 0
"""


def _render(template: str, *, data: str, **values: str) -> str:
    """Tokens first, the data LAST and once: a path is free to contain `@@`, and nothing inserted
    from the manifest is ever scanned for tokens again."""
    for name, value in values.items():
        template = template.replace(f"@@{name}@@", value)
    head, marker, tail = template.partition("@@DATA@@")
    assert marker and "@@" not in head + tail, "an unrendered token would reach the host"
    return head + data + tail


def _shell_safe_path(path: str) -> str:
    """Defence in depth before a path enters a script: the manifest is signed and parsed, and
    `store._check_materialisable` has refused control characters, but a path that reaches a
    shell must not depend on that having run. Refuses exactly what could escape the tree or break
    the tab- and newline-delimited framing; every other byte is carried by quoted expansions."""
    parsed = PurePosixPath(path)
    if (not path or parsed.is_absolute() or ".." in parsed.parts or path == MANIFEST_NAME
            or any(ord(char) < 0x20 or ord(char) == 0x7F for char in path)):
        raise StoreError(f"manifest path {path!r} cannot be passed to the host")
    return path


def _safe_url(url: str, schemes: Sequence[str]) -> str:
    """A URL that cannot break out of curl's quoted config line. The message never quotes it."""
    match = _URL.match(url)
    if match is None or match.group(1).lower() not in schemes or any(c in url for c in "\"'\\"):
        raise HostTransferError("a presigned URL was not in the expected form; nothing was sent")
    return url


@dataclass(frozen=True)
class _FetchResult:
    status: str
    size: int | None
    sha256: str | None
    code: int


def _records(output: str, kind: str) -> list[list[str]]:
    """The machine-readable lines, split on tabs. Lines that are not records (a login banner, a
    shell profile's chatter) are ignored; a record line that is malformed is never read as one.
    Split on `\\n` only: `str.splitlines` would also break a path at U+2028."""
    found = []
    for number, line in enumerate(output.split("\n"), start=1):
        if not line.startswith(_RECORD):
            continue
        fields = line.split("\t")
        if not fields[0].startswith(kind):
            raise HostTransferError(f"host output line {number} is an unexpected record")
        found.append(fields)
    return found


def _count(value: str, what: str) -> int:
    if not value.isdigit():
        raise HostTransferError(f"host output carries an unreadable {what}")
    return int(value)


class HostTrees:
    """A `store.TreeHost` over a `serve.Runner` — ssh to the serving host in production."""

    def __init__(self, run: Runner, trees: str, *, expires: float = PRESIGNED_TRANSFER_SECONDS,
                 parallelism: int = HOST_TRANSFER_PARALLELISM,
                 fetch_attempts: int = FETCH_ATTEMPTS, part_attempts: int = PART_ATTEMPTS,
                 pause_seconds: int = RETRY_PAUSE_SECONDS, schemes: Sequence[str] = ("https",),
                 settle_attempts: int = MOUNT_SETTLE_ATTEMPTS,
                 settle_seconds: float = MOUNT_SETTLE_SECONDS,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        trees = trees.rstrip("/") or "/"
        if not _TREES.match(trees) or ".." in PurePosixPath(trees).parts:
            raise ServeError(f"--serve-trees {trees!r} must be an absolute path of letters, digits "
                             f"and ._-/ with no '..'")
        if fetch_attempts < 2:
            raise ServeError("a fetch needs two attempts: a mismatch counts only when two agree")
        self.run = run
        self.trees = trees
        self.expires = float(expires)
        self.parallelism = int(parallelism)
        self.fetch_attempts = int(fetch_attempts)
        self.part_attempts = int(part_attempts)
        self.pause_seconds = int(pause_seconds)
        self.schemes = tuple(scheme.lower() for scheme in schemes)
        self.settle_attempts = int(settle_attempts)
        self.settle_seconds = float(settle_seconds)
        self.sleep = sleep

    # --- shared -----------------------------------------------------------------------------

    def _base(self, tree: Path) -> str:
        name = Path(tree).name
        if not _DIR_NAME.match(name):
            raise HostTransferError(f"tree directory name {name!r} is not a plain name")
        return f"{self.trees}/{name}"

    def _script(self, template: str, tree: Path, attempts: int, data: Sequence[str]) -> str:
        delimiter = f"THIRTYSPOKES_DATA_{secrets.token_hex(8)}"
        body = "".join(line + "\n" for line in data)
        assert delimiter not in body
        return _render(template, data=body, BASE=shlex.quote(self._base(tree)),
                       PARALLEL=str(self.parallelism), ATTEMPTS=str(attempts),
                       PAUSE=str(self.pause_seconds),
                       # curl's `--proto =https`: only these schemes, and no redirect elsewhere.
                       PROTO=shlex.quote("=" + ",".join(self.schemes)),
                       PARTIAL=shlex.quote(PARTIAL_MARKER), DELIM=delimiter)

    def _call(self, script: str) -> str:
        """The runner, with ONE translation: undecodable output is the host's failure. A runner
        error, a timeout and an OSError propagate as themselves."""
        try:
            return self.run(script)
        except UnicodeError as exc:
            raise HostTransferError("the host's output could not be decoded") from exc

    # --- fetch --------------------------------------------------------------------------------

    def fetch(self, bucket: S3Bucket, prefix: str, manifest: Manifest, dest: Path) -> None:
        """Place every manifest file under `<trees>/<dest.name>/` on the host; verify it here."""
        files = manifest.files
        for item in files:
            _shell_safe_path(item.path)
        data = [f"{item.size}\t{item.sha256}\t"
                f"{_safe_url(bucket.presign_get(prefix + item.path, expires=self.expires), self.schemes)}"
                f"\t{item.path}" for item in files]
        output = self._call(self._script(_FETCH_SCRIPT, dest, self.fetch_attempts, data))
        results = self._parse_fetch(output, files)
        self._judge_fetch(results, files)
        self._settle(dest, files)

    @staticmethod
    def _parse_fetch(output: str, files: Sequence[ManifestFile]) -> dict[int, _FetchResult]:
        results: dict[int, _FetchResult] = {}
        done: int | None = None
        for fields in _records(output, "THIRTYSPOKES-FETCH"):
            if fields[0] == "THIRTYSPOKES-FETCH-DONE" and len(fields) == 2 and done is None:
                done = _count(fields[1], "file count")
                continue
            if fields[0] != "THIRTYSPOKES-FETCH" or len(fields) != 7:
                raise HostTransferError("host output carries a malformed fetch record")
            index = _count(fields[1], "file index")
            status, size, digest, code, path = fields[2:]
            if (index >= len(files) or index in results or path != files[index].path
                    or status not in {"kept", "fetched", "mismatch", "failed", "unstable"}):
                raise HostTransferError(f"host output carries an unexpected record for file "
                                        f"{index}")
            known = status in {"kept", "fetched", "mismatch"}
            if known and not (size.isdigit() and _DIGEST.match(digest)):
                raise HostTransferError(f"host output carries an unreadable result for "
                                        f"{files[index].path}")
            results[index] = _FetchResult(status=status, size=int(size) if known else None,
                                          sha256=digest if known else None,
                                          code=_count(code, "exit code"))
        if done is None or done != len(files):
            raise HostTransferError("the host's fetch did not run to completion (no or a wrong "
                                    "completion line)")
        missing = [files[index].path for index in range(len(files)) if index not in results]
        if missing:
            raise HostTransferError(f"the host reported nothing for {missing}")
        return results

    @staticmethod
    def _judge_fetch(results: Mapping[int, _FetchResult], files: Sequence[ManifestFile]) -> None:
        """The verdict, here. Evidence first: a file shown to be the wrong bytes is a refusal even
        if another file's transfer failed in the same run, because those bytes will not change."""
        for index, item in enumerate(files):
            result = results[index]
            if (result.status == "mismatch" and result.size == item.size
                    and result.sha256 != item.sha256):
                raise StoreError(f"{item.path} hashes to {result.sha256}, the committed manifest "
                                 f"says {item.sha256}")
        for index, item in enumerate(files):
            result = results[index]
            if result.status == "mismatch":
                # The listing already showed the object at the committed size, so a different size
                # here is the transfer's; a matching size AND digest reported as a mismatch is a
                # host bug. Neither is evidence about the tree.
                raise HostTransferError(f"{item.path} arrived on the host at {result.size} bytes "
                                        f"where the bucket holds {item.size}: a transfer fault")
            if result.status in {"failed", "unstable"}:
                raise HostTransferError(
                    f"{item.path}: host transfer failed (curl exit {result.code})"
                    if result.status == "failed" else
                    f"{item.path}: two downloads on the host disagreed with each other")
            if result.size != item.size or result.sha256 != item.sha256:
                raise HostTransferError(f"the host reports {item.path} {result.status} at bytes "
                                        f"the manifest does not give; nothing is marked verified")

    def _settle(self, dest: Path, files: Sequence[ManifestFile]) -> None:
        """The controller's view of the tree must be what the host just wrote, read the way
        `admission.admit` reads it (a recursive listing, then each file) — or `--serve-trees` does
        not name the directory behind `<state>/trees`, and admission would judge the wrong tree."""
        expected = {item.path: item.size for item in files}
        for attempt in range(1, self.settle_attempts + 1):
            try:
                seen = ({path.relative_to(dest).as_posix(): path.stat().st_size
                         for path in dest.rglob("*") if path.is_file()}
                        if dest.is_dir() else None)
            except OSError:
                seen = None
            if seen == expected:
                return
            if attempt < self.settle_attempts:
                self.sleep(self.settle_seconds)
        raise HostTransferError(f"{dest} does not show the tree the host verified under "
                                f"{self._base(dest)}: --serve-trees does not name the directory "
                                f"behind <state>/trees, or the mount is stale")

    # --- promotion ----------------------------------------------------------------------------

    def upload(self, public: S3Bucket, destination: str, manifest: Manifest, tree: Path,
               pending: Sequence[ManifestFile]) -> None:
        """Publish `pending` from the verified tree on the host. Never a server-side copy.

        Each file becomes a multipart upload THIS process creates, with the `sha256` metadata fixed
        at creation; the host gets one presigned URL per part and nothing else. It re-hashes each
        file first and sends no part of one that does not match; each part it sends is reported
        with its MD5 and the ETag R2 returned. A file is completed only when its hash matches the
        manifest and every part's ETag is that part's MD5; every upload not completed is aborted.
        """
        pending = [item for item in pending]
        for item in pending:
            _shell_safe_path(item.path)
        # Uploads a killed controller left open under this prefix: billed until aborted, and never
        # completed by anyone, because only the process that created one holds its part list.
        for key, upload_id in public.open_multipart(destination):
            public.abort_multipart(key, upload_id)
        for item in pending:
            if item.size == 0:
                public.put(destination + item.path, b"", digest=item.sha256)
        sized = [item for item in pending if item.size > 0]
        if not sized:
            return
        opened: dict[int, str] = {}
        completed: set[int] = set()
        try:
            data = [f"F\t{item.size}\t{item.sha256}\t{item.path}" for item in sized]
            plans: dict[int, list[tuple[int, int, int]]] = {}
            for index, item in enumerate(sized):
                count = -(-item.size // PART_SIZE)
                if count > MAX_PARTS:
                    raise StoreError(f"{item.path} needs {count} parts, above S3's {MAX_PARTS}")
                key = destination + item.path
                opened[index] = public.create_multipart(key, digest=item.sha256)
                plans[index] = [(number, (number - 1) * PART_SIZE,
                                 min(PART_SIZE, item.size - (number - 1) * PART_SIZE))
                                for number in range(1, count + 1)]
                for number, offset, length in plans[index]:
                    url = _safe_url(public.presign_upload_part(key, opened[index], number,
                                                               expires=self.expires), self.schemes)
                    data.append(f"P\t{index}\t{number}\t{offset}\t{length}\t{url}")
            output = self._call(self._script(_UPLOAD_SCRIPT, tree, self.part_attempts, data))
            etags, failures = self._parse_upload(output, sized, plans)
            # Every file whose parts all check out is completed even when another file's did not:
            # it is the verified bytes under the committed metadata, and the next attempt then
            # resumes past it (`promote_submission`'s `in_place`) instead of re-sending it.
            for index, item in enumerate(sized):
                if index in etags:
                    public.complete_multipart(destination + item.path, opened[index], etags[index])
                    completed.add(index)
            if failures:
                raise failures[0]
        finally:
            for index, upload_id in opened.items():
                if index not in completed:
                    try:
                        public.abort_multipart(destination + sized[index].path, upload_id)
                    except Exception:         # noqa: BLE001 — the original failure is the news
                        pass

    @staticmethod
    def _parse_upload(output: str, files: Sequence[ManifestFile],
                      plans: Mapping[int, Sequence[tuple[int, int, int]]]
                      ) -> tuple[dict[int, list[tuple[int, str]]], list[HostTransferError]]:
        """(the part list of every file that may be completed, the failures of the rest).

        Output that cannot be read as a whole run, or a file whose bytes no longer hash to the
        manifest, raises at once and nothing is completed."""
        hashes: dict[int, tuple[str, str, str]] = {}
        parts: dict[tuple[int, int], tuple[str, str, str, str]] = {}
        done: tuple[int, int] | None = None
        for fields in _records(output, "THIRTYSPOKES-"):
            kind = fields[0]
            if kind == "THIRTYSPOKES-UPLOAD-DONE" and len(fields) == 3 and done is None:
                done = (_count(fields[1], "file count"), _count(fields[2], "part count"))
            elif kind == "THIRTYSPOKES-HASH" and len(fields) == 6:
                index = _count(fields[1], "file index")
                if index >= len(files) or index in hashes or fields[5] != files[index].path:
                    raise HostTransferError(f"host output carries an unexpected hash for file "
                                            f"{index}")
                hashes[index] = (fields[2], fields[3], fields[4])
            elif kind == "THIRTYSPOKES-PART" and len(fields) == 7:
                index, number = _count(fields[1], "file index"), _count(fields[2], "part number")
                if index not in plans or (index, number) in parts or not (
                        1 <= number <= len(plans[index])):
                    raise HostTransferError(f"host output carries an unexpected part {number} for "
                                            f"file {index}")
                parts[(index, number)] = (fields[3], fields[4], fields[5], fields[6])
            else:
                raise HostTransferError("host output carries a malformed upload record")
        if done != (len(files), sum(len(plan) for plan in plans.values())):
            raise HostTransferError("the host's upload did not run to completion (no or a wrong "
                                    "completion line)")
        # The judged bytes first: a file on the host that no longer hashes to the manifest is never
        # published, whatever happened to the parts.
        for index, item in enumerate(files):
            if index not in hashes:
                raise HostTransferError(f"the host reported no hash for {item.path}")
            status, size, digest = hashes[index]
            if status in {"missing", "mismatch"} or (status == "ok" and (
                    size != str(item.size) or digest != item.sha256)):
                raise StoreError(f"{item.path} on the host is {size} bytes hashing to {digest}, "
                                 f"the committed manifest says {item.size} at {item.sha256}; only "
                                 f"verified bytes are published")
            if status != "ok":
                raise HostTransferError(f"the host could not read {item.path}")
        etags: dict[int, list[tuple[int, str]]] = {}
        failures: list[HostTransferError] = []

        def part_etag(index: int, number: int) -> str:
            item = files[index]
            record = parts.get((index, number))
            if record is None:
                raise HostTransferError(f"the host reported nothing for part {number} of "
                                        f"{item.path}")
            status, md5, etag, code = record
            if status != "ok":
                raise HostTransferError(f"part {number} of {item.path} was not sent ({status}, "
                                        f"curl exit {code})")
            match = _ETAG.match(etag)
            if not _MD5.match(md5) or match is None or match.group(1).lower() != md5:
                raise HostTransferError(f"part {number} of {item.path} landed as bytes other than "
                                        f"those read from the tree (its ETag is not the part's "
                                        f"MD5)")
            return etag

        for index in range(len(files)):
            try:
                etags[index] = [(number, part_etag(index, number))
                                for number, _offset, _length in plans[index]]
            except HostTransferError as exc:
                failures.append(exc)
        return etags, failures


def host_trees(host: str, trees: str) -> HostTrees:
    """The daemon's wiring for `--serve-host` + `--serve-trees`: its own ssh runner, with the
    transfer's timeout rather than the serving runner's."""
    return HostTrees(ssh_runner(host, timeout=TREE_TRANSFER_TIMEOUT_SECONDS), trees)


__all__ = [
    "FETCH_ATTEMPTS", "HOST_TRANSFER_PARALLELISM", "HostTransferError", "HostTrees", "MAX_PARTS",
    "MOUNT_SETTLE_ATTEMPTS", "MOUNT_SETTLE_SECONDS", "PART_ATTEMPTS", "PRESIGNED_TRANSFER_SECONDS",
    "RETRY_PAUSE_SECONDS", "TREE_TRANSFER_TIMEOUT_SECONDS", "host_trees",
]
