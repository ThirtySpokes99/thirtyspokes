"""Owner: publish the measured runtime image to HuggingFace so miners can boot it.

Uploads the raw disk image plus a manifest carrying everything a miner needs to VERIFY it — the
image's sha256, the UKI hash, the dm-verity roothash, the MRTD/RTMR1/RTMR2 the validator gates on, and
the git commit of the recipe that built it.

The manifest is not a trust anchor; it is a convenience. The image is self-verifying: RTMR1 is pinned
on-chain by `orchestra-koth-owner`, and the build is reproducible, so a miner who fetches corrupted or
malicious bytes fails the validator's gate (`unapproved_runtime`) and earns nothing. Publishing the
hashes just lets them find that out *before* they burn a CVM boot — and lets them rebuild the image
themselves rather than trusting the owner at all.

    huggingface-cli login          # or export HF_TOKEN
    python scripts/publish_runtime_image.py \
        --version v14 --image /root/canon/koth-runtime-v14.tar.gz \
        --uki-sha256 4757aaab... --roothash 4af8c767... \
        --mrtd 9bf86e62... --rtmr1 6d8893eb... \
        --pool "meta-llama/llama-3.1-8b-instruct,openai/gpt-4o"
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess

from thirtyspokes.koth import imagestore
from thirtyspokes.koth.runtime import SUITE_VERSION, runtime_measurement

ZERO_RTMR = "00" * 48       # RTMR2 is zero under the direct-UKI boot the recipe uses


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recipe_commit() -> str:
    """The git commit of the recipe that produced this image — so a miner can check out exactly that
    revision and rebuild it, rather than trusting whatever `main` happens to hold."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_manifest(*, version: str, image_path: str, uki_sha256: str, roothash: str,
                   mrtd: str, rtmr1: str, rtmr2: str, pool: list[str],
                   bucket: str | None = None) -> dict:
    return {
        "schema": 1,
        "version": version,
        "bucket": bucket or imagestore.default_bucket(),
        "image": {
            "path": imagestore.image_path(version),
            "url": imagestore.public_url(imagestore.image_path(version), bucket=bucket),
            "sha256": sha256_file(image_path),
            "bytes": pathlib.Path(image_path).stat().st_size,
        },
        # what a miner rebuilds and compares against (the build is reproducible)
        "reproducible": {
            "recipe": "scripts/build_koth_image_prod.sh",
            "recipe_commit": recipe_commit(),
            "uki_sha256": uki_sha256,
            "verity_roothash": roothash,
        },
        # what the VALIDATOR gates on — these must match the owner's on-chain governance record
        "measurements": {
            "mrtd": mrtd,
            "rtmr1": rtmr1,
            "rtmr2": rtmr2,
            "note": "RTMR3 is derived from runtime+suite+pool and extended by the runtime at boot; "
                    "MRTD is GCP TDVF and identical on every GCP TDX guest, so RTMR1 is the "
                    "per-image anchor.",
        },
        "runtime_measurement": runtime_measurement(),
        "suite_version": SUITE_VERSION,
        "pool_allow_list": sorted(pool),
    }


def _refuse_unless_reproducible(dry_run: bool) -> str:
    """The manifest's whole value is `recipe_commit`: check out that revision, rebuild, confirm you
    got this image. If the working tree moved after the build, that claim is FALSE and the first
    miner to test it concludes the publisher is either careless or lying.

    This happened. v19 was published-ready with a recipe_commit naming source that had changed after
    the build (miner.py), and only a `--dry-run` plus a manual wheel comparison caught it. The check
    is mechanical, so it should not depend on remembering to look.
    """
    import subprocess
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                           check=True).stdout.strip()
    if dirty and not dry_run:
        raise SystemExit(
            "REFUSING TO PUBLISH: the working tree is dirty, so recipe_commit "
            f"{head[:12]} does not describe what was built.\n"
            f"Uncommitted:\n{dirty}\n"
            "Commit (or stash), REBUILD, re-measure on TDX, then publish.")
    return head


def _refuse_unless_head_built_this_image(build_dir: str | None, dry_run: bool) -> None:
    """Does HEAD actually describe the bytes in this image?

    `_refuse_unless_reproducible` only catches an UNCOMMITTED tree. It says nothing about a tree that
    moved on to further commits after the build — and that is the likelier mistake, because the build
    takes long enough that work continues while it runs. Measured on v23: the image was built at
    923e3a4, one more commit landed before publishing, and the manifest would have claimed a
    recipe_commit that rebuilds into a DIFFERENT image (different rootfs -> different roothash ->
    different RTMR1). A miner following the manifest would have been rejected `unapproved_runtime`
    and concluded the owner was lying.

    The build directory still holds the staged package that was baked in, so this is checkable rather
    than a matter of remembering: compare every .py under the image's site-packages/thirtyspokes
    against `git show HEAD:src/thirtyspokes/...`.
    """
    import subprocess
    if not build_dir:
        print("WARNING: --build-dir not given, so recipe_commit is UNVERIFIED. It is only a true "
              "claim if HEAD is exactly the commit this image was built from.", flush=True)
        return
    roots = sorted(pathlib.Path(build_dir).glob(
        "mkosi.extra/opt/koth/venv/lib/python3*/site-packages/thirtyspokes"))
    if not roots:
        raise SystemExit(f"REFUSING TO PUBLISH: no staged package under {build_dir} — "
                         "pass the build directory that produced this image.")
    root = roots[0]
    drifted = []
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root)
        r = subprocess.run(["git", "show", f"HEAD:src/thirtyspokes/{rel.as_posix()}"],
                           capture_output=True)
        if r.returncode != 0 or r.stdout != f.read_bytes():
            drifted.append(str(rel))
    if not drifted:
        return
    msg = (f"REFUSING TO PUBLISH: {len(drifted)} baked file(s) differ from HEAD, so recipe_commit "
           f"would name a commit that rebuilds into a DIFFERENT image:\n  "
           + "\n  ".join(drifted[:10])
           + ("\n  ..." if len(drifted) > 10 else "")
           + "\nEither rebuild at HEAD, or publish from a worktree pinned at the commit that built "
             "this image (`git worktree add <dir> <commit>`).")
    if dry_run:
        print(msg, flush=True)
        return
    raise SystemExit(msg)


def main() -> None:
    p = argparse.ArgumentParser(description="Publish the KOTH measured runtime image to HuggingFace")
    p.add_argument("--version", required=True, help="image version, e.g. v14")
    p.add_argument("--image", required=True, help="path to the raw disk tarball")
    p.add_argument("--bucket", help=f"HF bucket (default {imagestore.DEFAULT_BUCKET})")
    p.add_argument("--uki-sha256", required=True)
    p.add_argument("--roothash", required=True, help="dm-verity roothash")
    p.add_argument("--mrtd", required=True)
    p.add_argument("--rtmr1", required=True)
    p.add_argument("--rtmr2", default=ZERO_RTMR)
    p.add_argument("--pool", required=True, help="comma-separated owner-pinned model allow-list")
    p.add_argument("--build-dir", help="the build directory that produced this image (e.g. "
                                       "/root/koth-build-prod). Used to prove HEAD is the commit "
                                       "that built it — without it recipe_commit is unverified.")
    p.add_argument("--dry-run", action="store_true", help="print the manifest, upload nothing")
    args = p.parse_args()
    _refuse_unless_reproducible(args.dry_run)
    _refuse_unless_head_built_this_image(args.build_dir, args.dry_run)

    pool = [m.strip() for m in args.pool.split(",") if m.strip()]
    man = build_manifest(version=args.version, image_path=args.image, uki_sha256=args.uki_sha256,
                         roothash=args.roothash, mrtd=args.mrtd, rtmr1=args.rtmr1,
                         rtmr2=args.rtmr2, pool=pool, bucket=args.bucket)

    if args.dry_run:
        import json
        print(json.dumps(man, indent=2))
        return

    store = imagestore.RuntimeImageStore(args.bucket)
    print(f"uploading {man['image']['bytes'] / 1e6:.0f} MB to {store.bucket} ...", flush=True)
    store.publish(args.version, args.image, man)

    print("\npublished:")
    print(f"  bucket   {store.bucket}")
    print(f"  image    {imagestore.public_url(imagestore.image_path(args.version), bucket=store.bucket)}")
    print(f"  manifest {imagestore.public_url(imagestore.manifest_path(args.version), bucket=store.bucket)}")
    print(f"  sha256   {man['image']['sha256']}")
    print("\nA bucket has no immutable revision, so tell miners to check the sha256 above —")
    print("and remember RTMR1 on-chain is the gate that actually stops a bad image.")
    print("\nNow pin the SAME measurements on-chain (validators gate on these, not on the manifest):")
    print(f"  orchestra-koth-owner --mrtd {args.mrtd} --rtmr1 {args.rtmr1} --rtmr2 {args.rtmr2} \\")
    # ${{...}} escapes to a literal ${...}: these are SHELL variables for the operator to fill in,
    # not f-string fields. Unescaped, this raised NameError *after* a successful upload — making a
    # publish that had entirely worked look like it had failed.
    print(f"    --pool \"{','.join(pool)}\" --netuid ${{NETUID}} --wallet owner "
          f"--network ${{NETWORK}}")


if __name__ == "__main__":
    main()
