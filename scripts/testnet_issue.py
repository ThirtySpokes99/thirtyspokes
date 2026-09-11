#!/usr/bin/env python3
"""`thirtyspokes-owner issue`, against a MinIO bucket instead of Cloudflare R2 — the testnet rehearsal.

WHAT THIS STANDS IN FOR, AND THE ONE THING IT DOES NOT TEST. Production minting is
`access.scoped_credential`: a prefix-scoped R2 token that is a signed JWT Cloudflare accepts as a
session token, built with no network call. MinIO does not speak that JWT, so an `issue` against it
needs a different `mint` — and `Mailbox` takes `mint` as a parameter for exactly this reason (its
docstring: minting is the only step that touches the storage account's configuration). Everything
else on the path is the production code, unchanged: `registration_of` reads the uid and registration
block off the REAL chain, `Mailbox.issue` derives the prefix and enforces the one-shot ledger, the
envelope is sealed to the miner's ed25519 hotkey and signed by the owner's persisted mailbox key,
and `bucket.put` publishes it where `mailbox_key` says. The miner's `submit` then opens it with the
same `open_credential` it would use in production.

The stand-in mint is MinIO's STS `AssumeRole` with an inline session policy scoped to the
registration's prefix, so the credential it produces has the property R2's has — a miner cannot
write into another miner's submission — enforced by the store rather than promised by the client.
What is NOT exercised is the R2 JWT itself; that function has no offline test and gets none here.

Environment: `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` are the MinIO user that may assume roles
(NOT the root user — MinIO refuses AssumeRole for root), `R2_ENDPOINT` the http(s) origin.

    R2_ENDPOINT=http://127.0.0.1:9000 R2_ACCESS_KEY_ID=… R2_SECRET_ACCESS_KEY=… \\
      python scripts/testnet_issue.py --state /var/v3/state --bucket v3-testnet \\
        --netuid 526 --network test --wallet minirouter --owner-hotkey owner-hotkey \\
        --hotkey <miner ss58>
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from thirtyspokes.v3.access import AccessError, Credential, Mailbox
from thirtyspokes.v3.chain import BittensorChain
from thirtyspokes.v3.owner import CREDENTIAL_TTL_SECONDS, issue, mailbox_signer
from thirtyspokes.v3.store import r2_bucket


def sts_mint(*, endpoint: str, bucket: str, access_key_id: str, secret_access_key: str,
             ttl_seconds: int):
    """`(prefix) -> Credential` through MinIO's STS, scoped to that prefix by an inline policy."""
    import boto3

    def mint(prefix: str) -> Credential:
        if not prefix.endswith("/"):
            raise AccessError("a credential prefix must be slash-terminated")
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow",
                 "Action": ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload",
                            "s3:ListMultipartUploadParts"],
                 "Resource": [f"arn:aws:s3:::{bucket}/{prefix}*"]},
                # MinIO refuses an `s3:prefix` condition on ListBucketMultipartUploads, and the
                # store never lists in-progress uploads (`upload_tree` lists objects, then PUTs).
                {"Effect": "Allow", "Action": ["s3:ListBucket"],
                 "Resource": [f"arn:aws:s3:::{bucket}"],
                 "Condition": {"StringLike": {"s3:prefix": [f"{prefix}*"]}}},
            ],
        }
        sts = boto3.client("sts", endpoint_url=endpoint, aws_access_key_id=access_key_id,
                           aws_secret_access_key=secret_access_key, region_name="us-east-1")
        got = sts.assume_role(RoleArn="arn:minio:iam:::role/v3-submit", RoleSessionName="v3",
                              Policy=json.dumps(policy, separators=(",", ":")),
                              DurationSeconds=ttl_seconds)["Credentials"]
        return Credential(endpoint=endpoint, bucket=bucket, access_key_id=got["AccessKeyId"],
                          secret_access_key=got["SecretAccessKey"],
                          session_token=got["SessionToken"],
                          expires_at=got["Expiration"].astimezone(timezone.utc))
    return mint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--state", type=Path, required=True, metavar="DIR",
                        help="the validator's OWN --state directory (the one-shot ledger)")
    parser.add_argument("--hotkey", required=True, metavar="SS58", help="the miner's hotkey")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--network", default="test")
    parser.add_argument("--wallet", required=True, help="the owner's wallet name")
    parser.add_argument("--owner-hotkey", default="default")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--ttl-seconds", type=int, default=CREDENTIAL_TTL_SECONDS)
    args = parser.parse_args(argv)

    endpoint = os.environ["R2_ENDPOINT"]
    key_id, secret = os.environ["R2_ACCESS_KEY_ID"], os.environ["R2_SECRET_ACCESS_KEY"]
    parent = Credential(endpoint=endpoint, bucket=args.bucket, access_key_id=key_id,
                        secret_access_key=secret, session_token="",
                        expires_at=datetime.now(timezone.utc))
    mint = sts_mint(endpoint=endpoint, bucket=args.bucket, access_key_id=key_id,
                    secret_access_key=secret, ttl_seconds=args.ttl_seconds)
    chain = BittensorChain(netuid=args.netuid, wallet_name=args.wallet, network=args.network,
                           hotkey=args.owner_hotkey)
    mailbox = Mailbox(args.state / "mailbox.json", mailbox_signer(args.state), mint)
    try:
        print(issue(chain, mailbox, r2_bucket(parent).put, netuid=args.netuid, hotkey=args.hotkey))
        print(f"  owner key     {mailbox.owner_public_hex}   (the miner's --owner-key)")
    finally:
        chain.close()


if __name__ == "__main__":
    main()
