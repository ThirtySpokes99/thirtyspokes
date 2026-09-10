"""Pin the v3 reference model — the only sanctioned way to set the three digests in `config.py`.

`REFERENCE_REVISION`, `REFERENCE_CONFIG_HASH` and `REFERENCE_TOKENIZER_HASH` define the architecture
every miner's upload is checked against (§1.1). They are a ONE-WAY DOOR: rotating them invalidates
every model ever trained for the subnet at once. So they are derived by a script rather than pasted,
and this script re-derives them so anyone can check the pin rather than trust it.

WHY THE TOKENIZER DIGEST COVERS A SET, NOT A FILE. A tokenizer is `tokenizer.json` *and*
`vocab.json` *and* `merges.txt` *and* the two configs. Swapping any one of them changes what a
prompt tokenises to — and therefore what the Conductor sees — while a digest over `tokenizer.json`
alone stays green. The hash is taken over the sorted set, each file contributing its name and its
own digest, so a substitution anywhere in the set moves it.

NO WEIGHTS ARE DOWNLOADED. Only `config.json` and the tokenizer artefacts — a few MB against the
model's 71.9 GB. Pinning the architecture does not require possessing it.

    python scripts/pin_reference.py            # print the pin, compare against config.py
    python scripts/pin_reference.py --verify   # exit 1 if config.py disagrees
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from thirtyspokes.v3 import config as C  # noqa: E402


def derive(repo: str, revision: str | None = None) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.model_info(repo)
    rev = revision or info.sha

    cfg_path = hf_hub_download(repo, "config.json", revision=rev)
    config_hash = hashlib.sha256(pathlib.Path(cfg_path).read_bytes()).hexdigest()

    present = {f.rfilename for f in info.siblings}
    tok_files = sorted(f for f in C.REFERENCE_TOKENIZER_FILES if f in present)
    h = hashlib.sha256()
    for name in tok_files:
        h.update(name.encode())
        h.update(hashlib.sha256(pathlib.Path(hf_hub_download(repo, name, revision=rev)).read_bytes()).digest())

    cfg = json.loads(pathlib.Path(cfg_path).read_text())
    text = cfg.get("text_config", cfg)
    layers = int(text["num_hidden_layers"])
    interval = int(text.get("full_attention_interval", 1))
    attn_layers = layers // interval
    kv_bytes_per_token = 2 * attn_layers * int(text["num_key_value_heads"]) * int(text["head_dim"]) * 2

    return {
        "repo": repo, "revision": rev, "config_hash": config_hash,
        "tokenizer_hash": h.hexdigest(), "tokenizer_files": tok_files,
        "layers": layers, "full_attention_interval": interval, "attention_layers": attn_layers,
        "kv_bytes_per_token": kv_bytes_per_token,
        "multimodal": "vision_config" in cfg,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="derive and check the v3 reference pin")
    ap.add_argument("--repo", default=C.REFERENCE_MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--verify", action="store_true", help="exit 1 if config.py disagrees")
    args = ap.parse_args()

    d = derive(args.repo, args.revision)
    print(f"REFERENCE_MODEL          = {d['repo']!r}")
    print(f"REFERENCE_REVISION       = {d['revision']!r}")
    print(f"REFERENCE_CONFIG_HASH    = {d['config_hash']!r}")
    print(f"REFERENCE_TOKENIZER_HASH = {d['tokenizer_hash']!r}")
    print(f"  tokenizer files        : {d['tokenizer_files']}")
    print(f"  multimodal             : {d['multimodal']}  (vision tensors are part of the pin)")
    print(f"  {d['layers']} layers, full attention every {d['full_attention_interval']} "
          f"-> {d['attention_layers']} carry a growing KV cache")
    print(f"  KV                     : {d['kv_bytes_per_token'] / 1024:.0f} KB/token "
          f"({d['kv_bytes_per_token'] * 32768 / 1e9:.2f} GB per sequence at 32k)")

    if not args.verify:
        return
    bad = [(n, getattr(C, n), v) for n, v in
           (("REFERENCE_REVISION", d["revision"]),
            ("REFERENCE_CONFIG_HASH", d["config_hash"]),
            ("REFERENCE_TOKENIZER_HASH", d["tokenizer_hash"])) if getattr(C, n) != v]
    if bad:
        for name, pinned, derived in bad:
            print(f"\nMISMATCH {name}\n  config.py: {pinned}\n  derived  : {derived}", file=sys.stderr)
        print("\nThe pin is a ONE-WAY DOOR: a mismatch means either the upstream repo moved or "
              "config.py was edited by hand. Do not 'fix' config.py without deciding which.",
              file=sys.stderr)
        raise SystemExit(1)
    print("\nconfig.py matches the upstream artifact.")


if __name__ == "__main__":
    main()
