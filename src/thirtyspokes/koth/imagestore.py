"""HuggingFace bucket — where the owner publishes the measured runtime image.

Miners must boot the owner's measured image, so it has to be downloadable. It is served from a public
HF **bucket** (object storage, `hf://buckets/<ns>/<name>`): free, handles the ~560 MB disk, no egress
billed to the owner.

NOTE — a bucket is NOT a repo: there are no commit SHAs, so a miner cannot pin an immutable revision
the way they pin their own artifact. Integrity therefore rests on two things that do not depend on the
transport at all:

  * the manifest's `image.sha256` — check the bytes you downloaded;
  * **RTMR1, pinned on-chain** — the real gate. Boot the wrong bytes (corrupted, tampered, a mirror
    lying to you) and the validator rejects your proof `unapproved_runtime`; you earn nothing.

So the channel needs no trust, and the manifest is a convenience that lets a miner find out *before*
burning a CVM boot — plus the recipe commit, so they can rebuild the image instead of trusting the
owner at all.

Layout (our naming):
    runtime/<version>/koth-runtime.tar.gz     the raw disk image
    runtime/<version>/manifest.json           hashes + measurements + recipe commit
    runtime/latest.json                       which version is current
"""

from __future__ import annotations

import os
import pathlib
import shutil
import tempfile

DEFAULT_BUCKET = "thirtyspokes/cvm-runtime-image"
ENV_BUCKET = "THIRTYSPOKES_RUNTIME_BUCKET"

LATEST_PATH = "runtime/latest.json"


def image_path(version: str) -> str:
    return f"runtime/{version}/koth-runtime.tar.gz"


def manifest_path(version: str) -> str:
    return f"runtime/{version}/manifest.json"


def default_bucket() -> str:
    return os.environ.get(ENV_BUCKET) or DEFAULT_BUCKET


def bucket_uri(bucket: str | None = None) -> str:
    return f"hf://buckets/{bucket or default_bucket()}"


def public_url(path: str, *, bucket: str | None = None) -> str:
    """The plain https URL a miner can curl — no token, no SDK."""
    return f"https://huggingface.co/buckets/{bucket or default_bucket()}/resolve/{path}"


class RuntimeImageStore:
    """Owner-side publisher for the measured runtime image."""

    def __init__(self, bucket: str | None = None, token: str | None = None):
        self.bucket = bucket or default_bucket()
        self.token = token or os.environ.get("OWNER_HF_TOKEN") or os.environ.get("HF_TOKEN")

    def publish(self, version: str, image_file: str, manifest: dict, *, dry_run: bool = False):
        """Upload the image + manifest + latest pointer in one sync.

        Buckets are synced from a local tree, so we stage the exact layout and push it. `delete=False`:
        never wipe previously published versions — a miner may still be booting an older image.
        """
        import json

        from huggingface_hub import HfApi

        staged = pathlib.Path(tempfile.mkdtemp(prefix="koth_img_"))
        try:
            img_dst = staged / image_path(version)
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            # hard-link if we can (same fs) — copying 560 MB just to upload it is wasteful
            try:
                os.link(image_file, img_dst)
            except OSError:
                shutil.copy2(image_file, img_dst)
            (staged / manifest_path(version)).write_text(json.dumps(manifest, indent=2))
            (staged / LATEST_PATH).write_text(json.dumps({"version": version}, indent=2))

            api = HfApi(token=self.token)
            return api.sync_bucket(str(staged), bucket_uri(self.bucket),
                                   delete=False, dry_run=dry_run)
        finally:
            shutil.rmtree(staged, ignore_errors=True)

    def download(self, path: str, dest_dir: str) -> str:
        from huggingface_hub import HfApi
        api = HfApi(token=self.token)
        api.sync_bucket(f"{bucket_uri(self.bucket)}/{path}", dest_dir)
        return str(pathlib.Path(dest_dir) / pathlib.Path(path).name)
