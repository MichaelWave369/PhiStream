"""PocketRotor E3-P: offline, no-import Qwen3.5 HF runtime-source preflight.

Read a *locally installed* Python source file plus a local HF checkpoint's
config and weight-file metadata. Never import Transformers, Torch or a model,
never download weights, and never execute or modify a model.
Static AST matching is NOT proof of actual runtime hook execution.
"""
from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path

SCHEMA = "PHISTREAM_POCKETROTOR_E3_LOCAL_PREFLIGHT_V1"
SOURCE_NAME = "modeling_qwen3_5_moe.py"
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_CONFIG_BYTES = 512 * 1024
MAX_INDEX_BYTES = 4 * 1024 * 1024
MAX_LOCAL_FILES = 512


def require(ok: bool, detail: str) -> None:
    if not ok:
        raise ValueError(detail)


def read_bytes(path: str | Path, limit: int) -> bytes:
    target = Path(path)
    require(target.is_file() and not target.is_symlink(), "Missing or symlinked input")
    n = target.stat().st_size
    require(0 < n <= limit, "Input exceeds bounded source/config size")
    with target.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) == n, "Input changed or truncated during read")
    return raw


def json_unique(raw: bytes) -> dict:
    def no_dupe(pairs):
        out = {}
        for name, value in pairs:
            require(name not in out, "Duplicate JSON key")
            out[name] = value
        return out
    result = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=no_dupe,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )
    require(type(result) is dict, "JSON root must be an object")
    return result


def _attribute(node: ast.AST, prefix: str) -> bool:
    try:
        return ast.unparse(node) == prefix
    except Exception:
        return False


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    found = [node for node in cls.body if isinstance(node, ast.FunctionDef)
             and node.name == name]
    require(len(found) == 1, f"Missing unique {cls.name}.{name}")
    return found[0]


def _assigned_from(method: ast.FunctionDef, target_name: str, called_name: str) -> bool:
    for node in ast.walk(method):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if not isinstance(value, ast.Call) or not _attribute(value.func, called_name):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Tuple) and any(
                    isinstance(e, ast.Name) and e.id == target_name for e in target.elts
                ):
                    return True
                if isinstance(target, ast.Name) and target.id == target_name:
                    return True
    return False


def _router_returns_native_indices(forward: ast.FunctionDef) -> bool:
    for item in ast.walk(forward):
        if not isinstance(item, ast.Return) or not isinstance(item.value, ast.Tuple):
            continue
        parts = item.value.elts
        if len(parts) == 3 and all(isinstance(p, ast.Name) for p in parts):
            if [p.id for p in parts] == ["router_logits", "router_scores", "router_indices"]:
                return True
    return False


def inspect_source(raw: bytes) -> dict:
    """Check source syntax/structure, not arbitrary filename string matches."""
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=SOURCE_NAME)
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise ValueError("Transformers source is not parsable UTF-8 Python") from exc
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    required = ("Qwen3_5MoeTopKRouter", "Qwen3_5MoeExperts", "Qwen3_5MoeSparseMoeBlock")
    require(all(name in classes for name in required),
            "Required native HF Qwen3.5 router/experts/block classes missing")
    router = classes[required[0]]
    experts = classes[required[1]]
    block = classes[required[2]]
    router_init = _method(router, "__init__")
    router_forward = _method(router, "forward")
    expert_forward = _method(experts, "forward")
    block_init = _method(block, "__init__")
    block_forward = _method(block, "forward")

    router_names = [arg.arg for arg in router_forward.args.args]
    expert_names = [arg.arg for arg in expert_forward.args.args]
    require(router_names[:2] == ["self", "hidden_states"],
            "Router forward arguments changed")
    require(expert_names[:4] == [
        "self", "hidden_states", "top_k_index", "top_k_weights"
    ], "Experts forward argument contract changed")
    require(_router_returns_native_indices(router_forward)
            and _assigned_from(router_forward, "router_indices", "torch.topk"),
            "Router no longer returns native torch.topk selected IDs as third element")

    def constructor_attr(name: str, class_name: str) -> bool:
        for node in ast.walk(block_init):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if _attribute(node.value.func, class_name) and any(
                    _attribute(target, "self." + name) for target in node.targets
                ):
                    return True
        return False

    require(constructor_attr("gate", "Qwen3_5MoeTopKRouter")
            and constructor_attr("experts", "Qwen3_5MoeExperts"),
            "SparseMoEBlock no longer installs native gate and experts modules")
    gate_result = False
    experts_consumer = False
    for node in ast.walk(block_forward):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if _attribute(node.value.func, "self.gate"):
                if any(isinstance(target, ast.Tuple) and
                       len(target.elts) == 3 and
                       isinstance(target.elts[2], ast.Name) and
                       target.elts[2].id == "selected_experts"
                       for target in node.targets):
                    gate_result = True
            if _attribute(node.value.func, "self.experts"):
                args = node.value.args
                if len(args) >= 3 and isinstance(args[1], ast.Name) and args[1].id == "selected_experts":
                    experts_consumer = True
    require(gate_result and experts_consumer,
            "Block no longer passes third native gate result as second experts argument")
    # Attributes are assigned from config, not from unrelated metadata.
    attr = lambda value, name: any(
        isinstance(item, ast.Assign) and any(_attribute(t, "self." + name) for t in item.targets)
        and _attribute(item.value, "config." + value) for item in router_init.body
    )
    require(attr("num_experts", "num_experts")
            and attr("num_experts_per_tok", "top_k"),
            "Router num_experts/top_k config attributes changed")
    return {
        "source_structure_status": "STATIC_AST_CONTRACT_MATCH_ONLY",
        "router": required[0], "experts": required[1], "block": required[2],
        "router_tuple_position": 2, "experts_arg_position": 1,
        "source_sha256": sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "model_actually_executed": False,
        "module_hooks_tested_against_installed_runtime": False,
    }


def inspect_config(raw: bytes) -> dict:
    config = json_unique(raw)
    require(config.get("model_type") == "qwen3_5_moe",
            "Model must be Qwen3.5 MoE Hugging Face format, not GGUF")
    text = config.get("text_config")
    require(type(text) is dict and text.get("model_type") == "qwen3_5_moe_text",
            "Expected native Qwen3.5 MoE text config")
    required = {
        "num_hidden_layers": 40,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "hidden_size": 2048,
        "moe_intermediate_size": 512,
    }
    for key, value in required.items():
        require(type(text.get(key)) is int and text[key] == value,
                f"Checkpoint text_config {key} not supported in E3 pilot")
    layer_types = text.get("layer_types")
    require(isinstance(layer_types, list) and len(layer_types) == 40
            and layer_types.count("linear_attention") == 30
            and layer_types.count("full_attention") == 10,
            "Expected 30 linear-attention and 10 full-attention decoder layers")
    return {
        "config_status": "QWEN35_35B_A3B_TEXT_LAYOUT_MATCH_ONLY",
        "model_type": config["model_type"],
        "text_model_type": text["model_type"],
        "layers": 40, "experts": 256, "top_k": 8,
        "config_sha256": sha256(raw).hexdigest(),
        "weight_shapes_checked": False,
        "checkpoint_identity_authenticated": False,
    }


def inspect_local_directory(checkpoint: Path) -> dict:
    require(checkpoint.is_dir() and not checkpoint.is_symlink(),
            "Checkpoint must be an existing local directory")
    entries = list(checkpoint.iterdir())
    require(len(entries) <= MAX_LOCAL_FILES, "Too many checkpoint directory entries")
    for entry in entries:
        require(not entry.is_symlink(), "Symlinked checkpoint entries refused")
    gguf = [p for p in entries if p.is_file() and p.suffix.lower() == ".gguf"]
    safetensors = sorted(
        (p for p in entries if p.is_file() and p.suffix == ".safetensors"),
        key=lambda p: p.name,
    )
    if gguf:
        raise ValueError("GGUF checkpoint is not Hugging Face module format; E3 hook cannot run on Ollama")
    require(safetensors, "No local Hugging Face safetensors shards found")
    names = [p.name for p in safetensors]
    require(len(names) == len(set(names)), "Duplicate checkpoint weight files")
    size_bytes = sum(p.stat().st_size for p in safetensors)
    require(all(p.stat().st_size > 0 for p in safetensors),
            "Found empty model weights; cannot present as installed HF checkpoint")
    index = checkpoint / "model.safetensors.index.json"
    index_sha = None
    if index.exists():
        index_raw = read_bytes(index, MAX_INDEX_BYTES)
        decoded = json_unique(index_raw)
        weight_map = decoded.get("weight_map")
        require(isinstance(weight_map, dict) and weight_map,
                "Model weight index missing weight_map")
        referenced = set(weight_map.values())
        require(all(type(name) is str for name in referenced)
                and referenced == set(names),
                "Weight index shard references do not match local shard filenames")
        index_sha = sha256(index_raw).hexdigest()
    return {
        "weight_file_names_present": names,
        "weight_file_count": len(names),
        "weight_file_total_disk_bytes_unverified": size_bytes,
        "index_sha256": index_sha,
        "weight_file_contents_hashed": False,
        "weight_file_tensor_headers_checked": False,
        "weight_file_contents_authenticated": False,
    }


def preflight(checkpoint: str | Path, hf_source: str | Path) -> dict:
    """Never load weights, import transformers/torch, query CUDA or call network."""
    root = Path(checkpoint)
    require(Path(hf_source).name == SOURCE_NAME,
            "Specify installed modeling_qwen3_5_moe.py source path")
    source = inspect_source(read_bytes(hf_source, MAX_SOURCE_BYTES))
    config = inspect_config(read_bytes(root / "config.json", MAX_CONFIG_BYTES))
    local = inspect_local_directory(root)
    return {
        "schema": SCHEMA,
        "status": "STATIC_CONFIG_AND_SOURCE_MATCH_OPERATOR_REVIEW_ONLY",
        "source": source, "config": config, "local_weight_metadata": local,
        "model_execution": False, "model_weights_read": False,
        "transformers_imported": False, "torch_imported": False,
        "cuda_queried": False, "network_accessed": False,
        "real_native_routing_captured": False,
        "runtime_hook_execution_verified": False,
        "production_promoted": False,
        "operator_cautions": [
            "Static AST consistency cannot prove installed source was the executed code.",
            "No weight-file contents or tensor shapes were authenticated or read.",
            "No inference memory-fit assessment; 35B raw weights may exceed local RAM/VRAM.",
            "GGUF/Ollama and optimized kernels can bypass these HF module hooks.",
            "Only a separately approved, already-loaded, exact-runtime capture can produce native expert evidence.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only offline Qwen3.5 E3 compatibility preflight")
    parser.add_argument("--checkpoint", required=True, help="Existing HF safetensors checkpoint directory")
    parser.add_argument("--hf-source", required=True, help="Installed Transformers modeling_qwen3_5_moe.py")
    parser.add_argument("--output", help="Optional exclusive new report path; default stdout")
    args = parser.parse_args()
    result = preflight(args.checkpoint, args.hf_source)
    body = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        target = Path(args.output)
        require(target.parent.is_dir() and not target.is_symlink(),
                "Report output parent missing or symlinked")
        with target.open("x", encoding="utf-8") as stream:
            stream.write(body)
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
