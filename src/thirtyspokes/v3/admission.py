"""The §1.1 model-tree checks — the properties an adapter gave for free (the build plan M0 exits 8-13).

Under D11 a miner uploads ~70 GB of their own full weights instead of a delta over a pinned base.
An adapter made three things true *by construction*; this module is what makes them true again, and
each one is load-bearing:

  1. **No code execution.** `.bin` / `.pt` / `.pth` are pickle archives, and loading one runs
     arbitrary code ON THE OWNER'S VALIDATOR — the single machine that holds this subnet's scoring
     authority (D1). Refusing them is what carries the "no miner code ever executes" property (§2.1)
     through the switch to full-weight uploads, and §2.1 is what permits public benchmarks to be
     used at all. This check therefore runs FIRST, before anything in the tree is opened.

     THE PICKLE REFUSAL IS NECESSARY AND NOT SUFFICIENT (§1.1 point 2). A model tree carries a
     second execution path that has nothing to do with the weight format: the *custom-code* path. A
     `modeling_*.py` riding beside perfectly clean safetensors executes the moment the model is
     served with `trust_remote_code`, which is the ORDINARY way to load a non-upstream
     architecture — a serving stack that reaches for it is behaving normally, and no pickle was
     ever involved. So the same first-thing-we-do refuses any importable file in the tree, and the
     `config.json` check refuses the fields that ask for one. Refused, never stripped: silently
     deleting a `modeling_*.py` and loading the rest would serve an architecture the miner did not
     upload, which is a different failure and a worse one.
  2. **Architecture identity.** Without it, "the architecture is fixed" is a sentence in a document
     rather than a property of the system: a miner could upload a distilled 3B, a 70B, or a
     different family entirely. `config.json` is checked field-by-field, every tensor name and shape
     against the pinned inventory, and the tokenizer byte-for-byte.
  3. **Comparability.** Two models uploaded at different precisions are not the same experiment —
     quantisation changes quality — so a duel between them would compare numerics, not routing.

REFUSED, NEVER COERCED, AND NEVER LOADED. A hotkey gets one submission ever (§7), so a check that
repaired a near-miss would silently score a model other than the one submitted and spend the miner's
whole entry on it. Every refusal raises `AdmissionError` naming what mismatched, so the miner can
fix it and the operator can grep for it — the "Kind" property in its cheapest form. Note also what
this module never does: it reads safetensors *headers* only and never a byte of tensor data, so
admitting a 70 GB tree costs a few kilobytes of reads and there is no deserialiser anywhere on the
path.

WHY `config.json` IS COMPARED FIELD-BY-FIELD AND NOT BY ITS HASH. §1 has the owner publish a
`config.json` hash, which identifies the *reference* so anyone can confirm they hold the same one.
A miner's tree is a different question: re-saving a model with a newer library rewrites
`transformers_version` and key order without touching a single architectural decision, and refusing
that would spend a shot on cosmetics. So the pinned architectural fields are compared explicitly,
and the tensor inventory is the backstop for anything a field name would miss — no field can change
the shape of the model without changing a tensor.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ALLOWED_WEIGHT_SUFFIXES, REFUSED_WEIGHT_SUFFIXES, UNPINNED

# The architectural decisions §1 names — layer count, hidden size, expert count and routing width,
# vocabulary, RoPE settings — plus the identity fields, spelled as a Qwen3 MoE `config.json` spells
# them. A CLOSED LIST, not "every key in the file": the file also carries library versions, cache
# flags and generation defaults, none of which change the architecture, and refusing a tree over one
# of those would refuse a legal model. A field the reference has and the candidate lacks (or the
# reverse) is a mismatch just as much as a differing value.
ARCHITECTURAL_FIELDS = (
    "model_type",
    "architectures",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "moe_intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "num_experts",
    "num_experts_per_tok",
    "vocab_size",
    "max_position_embeddings",
    "rope_theta",
    # `rope_scaling` is the OLD spelling and this revision no longer uses it. Kept, because absent
    # on both sides still compares equal and an older tree would be checked by it; joined by the
    # current spelling, because §1 pins "RoPE settings" and none of these changes a tensor — so if
    # the closed list misses them, nothing catches them at all. Measured against the real artifact:
    # `rope_type` "default"->"yarn", `partial_rotary_factor` 0.25->1.0 and `mrope_section`
    # [11,11,10]->[16,8,8] were every one of them admitted before this line.
    "rope_scaling",
    "rope_type",
    "partial_rotary_factor",
    "mrope_section",
)

# The suffixes CPython's import machinery resolves a module NAME to — source, bytecode, and native
# extension (every ABI-tagged spelling of the last ends in `.so` or `.pyd`). That is the criterion
# rather than a list of scary file types: `auto_map` names a module, and these are the files that
# name can become. None of them has a legitimate place in a tree whose architecture is already
# pinned field-by-field, so refusing all of them costs no honest miner anything.
REFUSED_CODE_SUFFIXES = (".py", ".pyc", ".so", ".pyd")

# The `config.json` fields that ask for that code. A DENY-list here, unlike `ARCHITECTURAL_FIELDS`
# above, and the two are different questions: architecture identity is a COMPARISON against the
# reference (with the tensor inventory as its backstop — no field can change the shape of the model
# without changing a tensor), while this is a CAPABILITY the file requests, which nothing else in
# this module would notice. `custom_pipelines` is `auto_map` for pipelines, the identical mechanism.
#
# WHY THIS IS NOT REDUNDANT WITH `REFUSED_CODE_SUFFIXES`. `auto_map` also takes a
# `other-org/other-repo--modeling_x.Class` form, which loads code from a DIFFERENT repository — so
# there is no file in this tree to refuse and the field check is load-bearing on its own. The two
# checks close opposite halves of the same door.
CODE_ASKING_FIELDS = ("auto_map", "custom_pipelines", "trust_remote_code")

# `tokenizer.json` is the vocabulary, merges and normaliser in one file; `tokenizer_config.json`
# carries the special tokens and the chat template, which change what the same string tokenizes to.
# Both must be byte-identical or "the same prompt" is not the same sequence of ids.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")

# safetensors' own header ceiling. A file is untrusted miner input, and its first eight bytes are a
# length the validator would otherwise allocate: without this, `header_length = 2**63 - 1` is a
# one-line memory exhaustion attack on the machine that holds the scoring authority.
MAX_HEADER_BYTES = 100_000_000


class AdmissionError(Exception):
    """A tree that is not the pinned architecture. Always names what mismatched; never coerces."""


class _Absent:
    """Sentinel for 'this field is not in the file', so present-vs-absent reads as a mismatch."""

    def __repr__(self) -> str:
        return "<absent>"


_ABSENT = _Absent()


@dataclass(frozen=True)
class TensorSpec:
    """One tensor's identity: its safetensors dtype code (`BF16`) and its shape."""

    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Reference:
    """The pinned architecture as DATA — the owner's published inventory (§1), nothing more.

    Data rather than a loaded model is what makes these checks testable without the 70 GB artifact,
    and it is also what the owner publishes: a miner can run the identical check locally before
    spending their one shot.
    """

    config: Mapping[str, object]
    tensors: Mapping[str, TensorSpec]
    tokenizer_digest: str

    def __post_init__(self) -> None:
        # `config.UNPINNED` is a truthy sentinel precisely so that "not yet pinned" cannot pass for
        # "checked". Refusing it here means a caller that wires admission to the config module
        # before the owner publishes real digests fails loudly instead of admitting everything.
        if self.tokenizer_digest == UNPINNED:
            raise ValueError("reference tokenizer digest is still the UNPINNED sentinel")


def admit(root: Path, reference: Reference) -> None:
    """Run every §1.1 check over a model tree. Returns on success, raises `AdmissionError`.

    Order is deliberate. Both code refusals run before any file is opened — they read names, not
    bytes — because those are the checks protecting the validator itself rather than the fairness of
    a score. `config.json` comes next because a wrong architecture makes the tensor diff a wall of
    consequences rather than a cause, and a miner reading the refusal should see the cause; within
    it the custom-code fields are refused before the architecture is compared, for the same reason
    the file checks come first.
    """
    _refuse_pickle_archives(root)
    _refuse_bundled_code(root)
    _check_config(root, reference)
    _check_tensors(root, reference)
    _check_tokenizer(root, reference)


# The two that name the whole artifact rather than one tower inside it.
_OUTER_ONLY = ("model_type", "architectures")


# WHERE THOSE FIELDS ACTUALLY LIVE. The pinned reference is a multimodal config: `model_type` and
# `architectures` sit at the top level, and the other thirteen sit in a nested `text_config` block.
# Read at the top level they are ABSENT ON BOTH SIDES, so `_check_config` compared `_ABSENT` against
# `_ABSENT` thirteen times and passed every candidate — the field-by-field check §1.1 point 1 calls
# load-bearing was, on the real artifact, checking two fields of fifteen.
#
# The tensor inventory backstops most of what that missed: no field that changes a tensor's shape
# can pass it. What it does NOT backstop is the fields that change no tensor — `num_experts_per_tok`
# (the routing width §1 names explicitly), `max_position_embeddings`, `rope_theta`, `rope_scaling` —
# so a candidate could alter how the model routes and be served as uploaded.
#
# `scripts/pin_reference.py` already resolved the nesting this way; admission did not, which is
# how the two disagreed in silence. A candidate carrying no `text_config` is read at the top level
# and must then match the reference's values there, so flattening buys nothing: the values still
# have to be the reference's own.
# The blocks an architectural field can be spelled in, outermost first. Order matters twice: it is
# the precedence a loader applies, and it is the order `_architectural` reports a field's HOME in.
_BLOCKS: tuple[tuple[str, ...], ...] = ((), ("text_config",), ("text_config", "rope_parameters"))


def _at(config: Mapping[str, Any], path: tuple[str, ...]) -> Mapping[str, Any] | None:
    node: Any = config
    for step in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(step)
    return node if isinstance(node, Mapping) else None


def _architectural(config: Mapping[str, Any]) -> dict[str, Any]:
    """Each architectural field, keyed by the BLOCK it lives in rather than by its bare name.

    A flat view loses the one thing that decides what a loader builds. The pinned reference spells
    `num_experts_per_tok` inside `text_config`, and a candidate that hoists it to the top level with
    the reference's own value compared equal under a flat overlay — while `transformers` reads
    `text_config`, finds nothing there, and builds the text tower on a default. Admission passed a
    model that routes differently from the one it checked. Keying by path closes that: a field is
    compared where the reference keeps it, and a field that moved has moved.

    Innermost wins where a field appears twice — `partial_rotary_factor` is in both `text_config`
    and `rope_parameters` on the real artifact — because that is the copy a loader reads, and the
    other is then reported at its own key so neither goes unchecked.

    `model_type` and `architectures` name the WHOLE artifact and live at the top level only. They are
    read there and nowhere else: overlaying the text tower over them replaces the model's identity
    with its text half's, and a candidate declaring a different outer type stops being compared.
    """
    found: dict[str, Any] = {}
    for path in _BLOCKS:
        block = _at(config, path)
        if block is None:
            continue
        prefix = ".".join(path)
        for field in ARCHITECTURAL_FIELDS:
            if field in _OUTER_ONLY and path:
                continue                      # identity is an outer-level question, only
            if field in block:
                found[f"{prefix}.{field}" if prefix else field] = block[field]
    return found


def describe(root: Path) -> Reference:
    """Derive a `Reference` from a tree — the owner's pinning tool, run once on the reference model.

    It is also what makes M0 exit 13 checkable ("the reference model itself passes every check"):
    the checks must admit the thing they are defined against, and the honest way to test that is to
    define a reference from a tree and admit that same tree.
    """
    return Reference(
        # Kept whole: `_architectural` already selected the architectural fields and keyed them by
        # the block they live in, and re-filtering on bare names here would drop every nested one.
        config=_architectural(_config(root)),
        tensors=inventory(root),
        tokenizer_digest=tokenizer_digest(root))


def inventory(root: Path) -> dict[str, TensorSpec]:
    """Every tensor in the tree, from safetensors headers only — no tensor data is ever read."""
    shards = sorted(path for path in root.rglob("*")
                    if path.is_file() and path.suffix in ALLOWED_WEIGHT_SUFFIXES)
    if not shards:
        raise AdmissionError(
            f"no weight file: the tree holds nothing ending in "
            f"{'/'.join(ALLOWED_WEIGHT_SUFFIXES)}")

    tensors: dict[str, TensorSpec] = {}
    for shard in shards:
        for name, spec in _header(shard, root).items():
            if name in tensors:
                raise AdmissionError(
                    f"duplicate tensor {name!r}: it appears in more than one shard, so which "
                    f"weights would be scored is ambiguous (seen again in {shard.name})")
            tensors[name] = spec
    return tensors


def tokenizer_digest(root: Path) -> str:
    """One digest over every tokenizer file, in a pinned order.

    The file NAME and its LENGTH are hashed alongside the bytes so that renaming a tokenizer file,
    or moving a byte across the boundary between two of them, changes the digest. A bare
    concatenation would not notice either.
    """
    digest = hashlib.sha256()
    for name in TOKENIZER_FILES:
        path = root / name
        if not path.is_file():
            raise AdmissionError(f"tokenizer file {name!r} is missing from the tree")
        blob = path.read_bytes()
        digest.update(name.encode())
        digest.update(len(blob).to_bytes(8, "little"))
        digest.update(blob)
    return digest.hexdigest()


def _refuse_pickle_archives(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in REFUSED_WEIGHT_SUFFIXES:
            raise AdmissionError(
                f"pickle archive refused: {path.relative_to(root)} — "
                f"{'/'.join(REFUSED_WEIGHT_SUFFIXES)} run arbitrary code when loaded, and this "
                f"validator holds the subnet's scoring authority. Upload safetensors only")


def _refuse_bundled_code(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in REFUSED_CODE_SUFFIXES:
            raise AdmissionError(
                f"bundled code refused: {path.relative_to(root)} — a tree carrying "
                f"{'/'.join(REFUSED_CODE_SUFFIXES)} is a tree asking to be loaded with "
                f"trust_remote_code, which runs the miner's file on the validator that holds the "
                f"subnet's scoring authority. The architecture is pinned, so no legitimate tree "
                f"needs custom code. Upload weights, config and tokenizer only")


def _refuse_custom_code_fields(config: Mapping[str, object]) -> None:
    """Recursive, because `auto_map` is read off sub-configs as well as off the top level: "anywhere
    in the file" is the property here exactly as "anywhere in the tree" is for the files."""
    for field, value in config.items():
        if field in CODE_ASKING_FIELDS:
            raise AdmissionError(
                f"custom-code config refused: config.json carries {field!r}, the field that asks "
                f"for a python module to be imported when the model is loaded. It can name code in "
                f"another repository, so refusing the files in this tree is not enough. The "
                f"architecture is pinned; a tree that needs custom code is not the reference")
        if isinstance(value, Mapping):
            _refuse_custom_code_fields(value)


def _check_config(root: Path, reference: Reference) -> None:
    raw = _config(root)
    # The deny-list is asked of the WHOLE file: `auto_map` and `trust_remote_code` are top-level
    # requests for code, and nesting is where the architectural fields live, not where capability
    # lives.
    _refuse_custom_code_fields(raw)
    config = _architectural(raw)
    # Over the UNION of both sides' keys, so a field the candidate added where the reference has
    # none is a mismatch too — that is the hoisting case, and comparing only the reference's keys
    # would let it through from the other direction.
    for field in sorted(set(config) | set(reference.config)):
        want = reference.config.get(field, _ABSENT)
        got = config.get(field, _ABSENT)
        if got != want:
            raise AdmissionError(
                f"architecture mismatch: config.json field {field!r} is {got!r}, "
                f"the pinned reference has {want!r}. If you did not mean to change the "
                f"architecture, this is usually `save_pretrained` re-serialising config.json under a "
                f"different transformers version — copy the reference tree's own config.json back "
                f"and re-run `check`.")


def _check_tensors(root: Path, reference: Reference) -> None:
    found = inventory(root)
    # Sorted, so the refusal a miner sees for a given tree is always the same one. A renamed tensor
    # is both an extra and a missing; whichever sorts first is the message, and either is true.
    for name in sorted(set(found) | set(reference.tensors)):
        want = reference.tensors.get(name)
        got = found.get(name)
        if want is None:
            raise AdmissionError(f"extra tensor: {name!r} is not in the pinned inventory")
        if got is None:
            raise AdmissionError(f"missing tensor: {name!r} is in the pinned inventory but not "
                                 f"in this tree")
        if got.shape != want.shape:
            raise AdmissionError(f"reshaped tensor: {name!r} has shape {got.shape}, the pinned "
                                 f"inventory has {want.shape}")
        if got.dtype != want.dtype:
            raise AdmissionError(f"wrong dtype: {name!r} is {got.dtype}, the pinned inventory has "
                                 f"{want.dtype} — two precisions are not the same experiment")


def _check_tokenizer(root: Path, reference: Reference) -> None:
    got = tokenizer_digest(root)
    if got != reference.tokenizer_digest:
        raise AdmissionError(
            f"tokenizer is not byte-identical to the reference: {got} != "
            f"{reference.tokenizer_digest}")


def _config(root: Path) -> dict:
    path = root / "config.json"
    if not path.is_file():
        raise AdmissionError("config.json is missing: the architecture cannot be verified")
    try:
        config = json.loads(path.read_bytes())
    except ValueError as exc:
        raise AdmissionError(f"config.json is not valid JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise AdmissionError(f"config.json is not valid JSON: expected an object, got "
                             f"{type(config).__name__}")
    return config


def _header(shard: Path, root: Path) -> dict[str, TensorSpec]:
    """The safetensors header: eight little-endian length bytes, then that many bytes of JSON.

    Parsed here rather than through the `safetensors` package because the package's readers open a
    file *into a framework* (torch, numpy) — they materialise tensors, which is both the thing this
    module must not do to untrusted input and impossible for bf16 without a deep learning stack.
    The format's header is a documented, stable prefix; reading it is the narrow seam.
    """
    rel = shard.relative_to(root)
    with shard.open("rb") as handle:
        size = handle.read(8)
        if len(size) != 8:
            raise AdmissionError(f"malformed weight file {rel}: too short to hold a header length")
        length = int.from_bytes(size, "little")
        if not 0 < length <= MAX_HEADER_BYTES:
            raise AdmissionError(f"malformed weight file {rel}: declares a {length}-byte header, "
                                 f"outside 1..{MAX_HEADER_BYTES}")
        blob = handle.read(length)
    if len(blob) != length:
        raise AdmissionError(f"malformed weight file {rel}: header runs past the end of the file")
    try:
        header = json.loads(blob)
        return {name: TensorSpec(dtype=entry["dtype"], shape=tuple(entry["shape"]))
                for name, entry in header.items() if name != "__metadata__"}
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise AdmissionError(f"malformed weight file {rel}: unreadable safetensors header "
                             f"({exc!r})") from exc
