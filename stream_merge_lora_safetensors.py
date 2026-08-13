#!/usr/bin/env python3
"""Out-of-core merge of one PEFT LoRA into sharded Safetensors weights.

The full Transformers model is never instantiated. The merger validates the
adapter against the checkpoint index, processes one base shard at a time, and
atomically writes a standard Hugging Face checkpoint with the original shard
layout.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

VERSION = "1.0.0"
LORA_KEY_RE = re.compile(r"^(?P<module>.+)\.lora_(?P<kind>A|B)(?:\.[^.]+)?\.weight$")
MODEL_SHARD_RE = re.compile(r"^model-\d+-of-\d+\.safetensors$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream one LoRA into a sharded Safetensors checkpoint."
    )
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row-chunk", type=int, default=2048)
    parser.add_argument(
        "--threads", type=int, default=min(64, max(1, os.cpu_count() or 1))
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate without creating output files.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild already completed output shards."
    )
    parser.add_argument(
        "--hardlink-untouched",
        action="store_true",
        help="Hard-link untouched shards instead of copying them. Disabled by default for independence.",
    )
    parser.add_argument(
        "--safe-merge",
        action="store_true",
        help="Check every merged tensor for NaN/Inf.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temp_path, path)


def strip_peft_prefix(module_name: str) -> str:
    for prefix in ("base_model.model.", "base_model."):
        if module_name.startswith(prefix):
            return module_name[len(prefix) :]
    return module_name


def load_lora_scale(adapter_config: dict[str, Any]) -> float:
    if adapter_config.get("peft_type") != "LORA":
        raise ValueError("Only PEFT LoRA adapters are supported.")

    unsupported = {
        "use_dora": adapter_config.get("use_dora", False),
        "fan_in_fan_out": adapter_config.get("fan_in_fan_out", False),
        "modules_to_save": bool(adapter_config.get("modules_to_save")),
        "rank_pattern": bool(adapter_config.get("rank_pattern")),
        "alpha_pattern": bool(adapter_config.get("alpha_pattern")),
        "bias": adapter_config.get("bias", "none") != "none",
    }
    enabled = [name for name, value in unsupported.items() if value]
    if enabled:
        raise ValueError("Unsupported adapter features: " + ", ".join(enabled))

    rank = int(adapter_config["r"])
    alpha = float(adapter_config["lora_alpha"])
    if rank <= 0:
        raise ValueError(f"Invalid LoRA rank: {rank}")

    return (
        alpha / math.sqrt(rank)
        if adapter_config.get("use_rslora", False)
        else alpha / rank
    )


def discover_lora_pairs(adapter_file: Path) -> dict[str, dict[str, str]]:
    pairs: dict[str, dict[str, str]] = {}
    unmatched: list[str] = []

    with safe_open(str(adapter_file), framework="pt", device="cpu") as reader:
        for adapter_key in reader.keys():  # noqa: SIM118 - safe_open is not iterable.
            match = LORA_KEY_RE.match(adapter_key)
            if match is None:
                unmatched.append(adapter_key)
                continue

            module_name = strip_peft_prefix(match.group("module"))
            base_key = f"{module_name}.weight"
            kind = match.group("kind")
            if kind in pairs.setdefault(base_key, {}):
                raise RuntimeError(f"Duplicate LoRA {kind} tensor for {base_key}")
            pairs[base_key][kind] = adapter_key

    if unmatched:
        raise RuntimeError(f"Found unsupported adapter tensor keys: {unmatched[:10]}")
    if not pairs:
        raise RuntimeError(f"No LoRA A/B tensors found in {adapter_file}")

    incomplete = {
        key: value for key, value in pairs.items() if set(value) != {"A", "B"}
    }
    if incomplete:
        raise RuntimeError(f"Incomplete LoRA pairs: {list(incomplete.items())[:10]}")

    return pairs


def validate_paths(
    base_dir: Path, adapter_dir: Path, output_dir: Path
) -> tuple[Path, Path, Path]:
    index_file = base_dir / "model.safetensors.index.json"
    adapter_config_file = adapter_dir / "adapter_config.json"
    adapter_file = adapter_dir / "adapter_model.safetensors"

    for required in (index_file, adapter_config_file, adapter_file):
        if not required.is_file():
            raise FileNotFoundError(required)
    if base_dir == output_dir:
        raise ValueError("Output directory must differ from the base directory.")
    if output_dir == adapter_dir or output_dir in adapter_dir.parents:
        raise ValueError("Output directory must differ from the adapter directory.")

    incomplete = list(
        (base_dir / ".cache" / "huggingface" / "download").rglob("*.incomplete")
    )
    if incomplete:
        raise RuntimeError(
            f"The base model download has {len(incomplete)} incomplete file(s)."
        )

    return index_file, adapter_config_file, adapter_file


def build_shard_map(weight_map: dict[str, str]) -> dict[str, list[str]]:
    shard_to_keys: dict[str, list[str]] = defaultdict(list)
    for tensor_key, shard_name in weight_map.items():
        shard_to_keys[shard_name].append(tensor_key)
    for keys in shard_to_keys.values():
        keys.sort()
    return dict(shard_to_keys)


def validate_checkpoint_and_adapter(
    base_dir: Path,
    adapter_file: Path,
    adapter_config: dict[str, Any],
    weight_map: dict[str, str],
    shard_to_keys: dict[str, list[str]],
    lora_pairs: dict[str, dict[str, str]],
) -> dict[str, Any]:
    missing_base_keys = sorted(set(lora_pairs) - set(weight_map))
    if missing_base_keys:
        raise RuntimeError(
            f"Adapter keys missing from base index: {missing_base_keys[:20]}"
        )

    missing_shards = [
        name for name in sorted(shard_to_keys) if not (base_dir / name).is_file()
    ]
    empty_shards = [
        name
        for name in sorted(shard_to_keys)
        if (base_dir / name).is_file() and (base_dir / name).stat().st_size == 0
    ]
    if missing_shards or empty_shards:
        raise RuntimeError(
            f"Incomplete base checkpoint: missing={missing_shards[:10]}, empty={empty_shards[:10]}"
        )

    touched_by_shard: dict[str, list[str]] = defaultdict(list)
    for base_key in lora_pairs:
        touched_by_shard[weight_map[base_key]].append(base_key)

    configured_rank = int(adapter_config["r"])
    base_dtypes: set[str] = set()
    adapter_dtypes: set[str] = set()
    shape_errors: list[str] = []

    with safe_open(str(adapter_file), framework="pt", device="cpu") as adapter_reader:
        for shard_name, touched_keys in sorted(touched_by_shard.items()):
            with safe_open(
                str(base_dir / shard_name), framework="pt", device="cpu"
            ) as base_reader:
                actual_keys = set(base_reader.keys())
                expected_keys = set(shard_to_keys[shard_name])
                if actual_keys != expected_keys:
                    raise RuntimeError(
                        f"Index/shard mismatch in {shard_name}: "
                        f"missing={sorted(expected_keys - actual_keys)[:10]}, "
                        f"extra={sorted(actual_keys - expected_keys)[:10]}"
                    )

                for base_key in touched_keys:
                    pair = lora_pairs[base_key]
                    base_slice = base_reader.get_slice(base_key)
                    a_slice = adapter_reader.get_slice(pair["A"])
                    b_slice = adapter_reader.get_slice(pair["B"])
                    base_shape = tuple(base_slice.get_shape())
                    a_shape = tuple(a_slice.get_shape())
                    b_shape = tuple(b_slice.get_shape())
                    base_dtypes.add(str(base_slice.get_dtype()))
                    adapter_dtypes.update(
                        (str(a_slice.get_dtype()), str(b_slice.get_dtype()))
                    )

                    if len(base_shape) != 2 or len(a_shape) != 2 or len(b_shape) != 2:
                        shape_errors.append(
                            f"{base_key}: base={base_shape}, A={a_shape}, B={b_shape}"
                        )
                        continue
                    if a_shape[0] != configured_rank or b_shape[1] != configured_rank:
                        shape_errors.append(
                            f"{base_key}: configured rank={configured_rank}, A={a_shape}, B={b_shape}"
                        )
                        continue
                    if b_shape[1] != a_shape[0] or base_shape != (
                        b_shape[0],
                        a_shape[1],
                    ):
                        shape_errors.append(
                            f"{base_key}: base={base_shape}, A={a_shape}, B={b_shape}"
                        )

    if shape_errors:
        raise RuntimeError(
            "LoRA/base shape validation failed: " + "; ".join(shape_errors[:10])
        )

    base_bytes = sum((base_dir / name).stat().st_size for name in shard_to_keys)
    return {
        "base_shards": len(shard_to_keys),
        "base_bytes": base_bytes,
        "lora_pairs": len(lora_pairs),
        "touched_shards": len(touched_by_shard),
        "untouched_shards": len(shard_to_keys) - len(touched_by_shard),
        "base_dtypes": sorted(base_dtypes),
        "adapter_dtypes": sorted(adapter_dtypes),
    }


def merge_linear_weight_peft_equivalent(
    base: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
    row_chunk: int,
    safe_merge: bool,
) -> torch.Tensor:
    """Match PEFT's CPU merge order: FP32 B@A -> base dtype -> base add."""
    if row_chunk <= 0:
        raise ValueError("row_chunk must be greater than zero.")
    if base.ndim != 2 or lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError(
            f"Expected 2D tensors: base={base.shape}, A={lora_a.shape}, B={lora_b.shape}"
        )
    if (
        tuple(base.shape) != (lora_b.shape[0], lora_a.shape[1])
        or lora_b.shape[1] != lora_a.shape[0]
    ):
        raise ValueError(
            f"Shape mismatch: base={base.shape}, A={lora_a.shape}, B={lora_b.shape}"
        )

    merged = torch.empty_like(base)
    a_fp32 = lora_a.float().contiguous()

    for row_start in range(0, base.shape[0], row_chunk):
        row_end = min(row_start + row_chunk, base.shape[0])
        b_fp32 = lora_b[row_start:row_end].float().contiguous()
        delta = torch.mm(b_fp32, a_fp32).mul_(scale).to(dtype=base.dtype)
        merged[row_start:row_end].copy_(base[row_start:row_end] + delta)
        del b_fp32, delta

    if safe_merge and not torch.isfinite(merged).all():
        raise ValueError("NaN or Inf detected in merged weight.")
    return merged.contiguous()


def copy_model_metadata(
    base_dir: Path, output_dir: Path, shard_names: set[str]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in base_dir.iterdir():
        if (
            not source.is_file()
            or source.name in shard_names
            or source.name == "model.safetensors.index.json"
        ):
            continue
        shutil.copy2(source, output_dir / source.name)


def verify_shard_header(
    output_shard: Path, base_shard: Path, expected_keys: list[str]
) -> None:
    with (
        safe_open(str(base_shard), framework="pt", device="cpu") as base_reader,
        safe_open(str(output_shard), framework="pt", device="cpu") as output_reader,
    ):
        if set(output_reader.keys()) != set(expected_keys):
            raise RuntimeError(f"Output shard key mismatch: {output_shard}")
        for key in expected_keys:
            source = base_reader.get_slice(key)
            output = output_reader.get_slice(key)
            if (
                source.get_shape() != output.get_shape()
                or source.get_dtype() != output.get_dtype()
            ):
                raise RuntimeError(
                    f"Output tensor metadata mismatch: {output_shard}:{key}"
                )


def materialize_untouched_shard(source: Path, temp_path: Path, hardlink: bool) -> None:
    if temp_path.exists():
        temp_path.unlink()
    if hardlink:
        try:
            os.link(source, temp_path)
            return
        except OSError:
            pass
    shutil.copy2(source, temp_path)


def prepare_output(
    base_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    index_file: Path,
    shard_names: set[str],
    plan: dict[str, Any],
    scale: float,
) -> tuple[Path, dict[str, Any]]:
    marker_path = output_dir / "merge_info.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not marker_path.is_file():
            raise RuntimeError(
                f"Non-empty output has no merge marker; refusing to reuse: {output_dir}"
            )
        marker = read_json(marker_path)
        if marker.get("base_dir") != str(base_dir) or marker.get("adapter_dir") != str(
            adapter_dir
        ):
            raise RuntimeError(
                "Existing output was created for a different base model or adapter."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        marker = {}

    copy_model_metadata(base_dir, output_dir, shard_names)
    shutil.copy2(index_file, output_dir / index_file.name)

    marker.update(
        {
            "stream_merger_version": VERSION,
            "status": "running",
            "base_dir": str(base_dir),
            "adapter_dir": str(adapter_dir),
            "output_dir": str(output_dir),
            "lora_scale": scale,
            "compute": "PEFT-equivalent CPU merge: FP32 B@A, cast delta to base dtype, then add",
            **plan,
        }
    )
    marker.setdefault("completed_shards", [])
    write_json_atomic(marker_path, marker)
    return marker_path, marker


def print_plan(
    plan: dict[str, Any],
    scale: float,
    base_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
) -> None:
    print("Preflight passed", flush=True)
    print(f"  base: {base_dir}", flush=True)
    print(f"  adapter: {adapter_dir}", flush=True)
    print(f"  output: {output_dir}", flush=True)
    print(f"  base shards: {plan['base_shards']}", flush=True)
    print(
        f"  touched/untouched shards: {plan['touched_shards']}/{plan['untouched_shards']}",
        flush=True,
    )
    print(f"  LoRA pairs: {plan['lora_pairs']}", flush=True)
    print(f"  LoRA scale: {scale}", flush=True)
    print(f"  base size: {plan['base_bytes'] / 2**40:.2f} TiB", flush=True)
    print(
        f"  base/adapter dtypes: {plan['base_dtypes']}/{plan['adapter_dtypes']}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    adapter_dir = args.adapter_dir.resolve()
    output_dir = args.output_dir.resolve()

    index_file, adapter_config_file, adapter_file = validate_paths(
        base_dir, adapter_dir, output_dir
    )
    adapter_config = read_json(adapter_config_file)
    scale = load_lora_scale(adapter_config)
    lora_pairs = discover_lora_pairs(adapter_file)
    index = read_json(index_file)
    weight_map: dict[str, str] = index["weight_map"]
    shard_to_keys = build_shard_map(weight_map)
    plan = validate_checkpoint_and_adapter(
        base_dir, adapter_file, adapter_config, weight_map, shard_to_keys, lora_pairs
    )
    print_plan(plan, scale, base_dir, adapter_dir, output_dir)

    if args.check_only:
        print("Check completed; no output was created.", flush=True)
        return

    torch.set_num_threads(max(1, args.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    shard_names = set(shard_to_keys)
    marker_path, marker = prepare_output(
        base_dir, adapter_dir, output_dir, index_file, shard_names, plan, scale
    )
    completed = set(marker.get("completed_shards", []))
    pair_keys = set(lora_pairs)

    with (
        torch.inference_mode(),
        safe_open(str(adapter_file), framework="pt", device="cpu") as adapter_reader,
    ):
        for shard_index, shard_name in enumerate(sorted(shard_to_keys), start=1):
            source_shard = base_dir / shard_name
            output_shard = output_dir / shard_name
            temp_shard = output_dir / f".{shard_name}.tmp"
            shard_keys = shard_to_keys[shard_name]
            touched_keys = pair_keys.intersection(shard_keys)

            if output_shard.exists() and not args.force:
                verify_shard_header(output_shard, source_shard, shard_keys)
                print(
                    f"[{shard_index}/{len(shard_to_keys)}] verified existing {shard_name}",
                    flush=True,
                )
                completed.add(shard_name)
                continue

            if not touched_keys:
                print(
                    f"[{shard_index}/{len(shard_to_keys)}] copy unchanged {shard_name}",
                    flush=True,
                )
                materialize_untouched_shard(
                    source_shard, temp_shard, args.hardlink_untouched
                )
            else:
                print(
                    f"[{shard_index}/{len(shard_to_keys)}] merge {shard_name}: {len(touched_keys)} tensors",
                    flush=True,
                )
                if temp_shard.exists():
                    temp_shard.unlink()

                with safe_open(
                    str(source_shard), framework="pt", device="cpu"
                ) as base_reader:
                    output_tensors: dict[str, torch.Tensor] = {}
                    metadata = base_reader.metadata()
                    for tensor_key in base_reader.keys():  # noqa: SIM118 - safe_open is not iterable.
                        base_tensor = base_reader.get_tensor(tensor_key)
                        pair = lora_pairs.get(tensor_key)
                        if pair is None:
                            output_tensors[tensor_key] = base_tensor
                            continue

                        lora_a = adapter_reader.get_tensor(pair["A"])
                        lora_b = adapter_reader.get_tensor(pair["B"])
                        output_tensors[tensor_key] = (
                            merge_linear_weight_peft_equivalent(
                                base_tensor,
                                lora_a,
                                lora_b,
                                scale,
                                args.row_chunk,
                                args.safe_merge,
                            )
                        )
                        del lora_a, lora_b

                    save_file(output_tensors, str(temp_shard), metadata=metadata)

                del output_tensors
                gc.collect()

            verify_shard_header(temp_shard, source_shard, shard_keys)
            os.replace(temp_shard, output_shard)
            completed.add(shard_name)
            marker["completed_shards"] = sorted(completed)
            write_json_atomic(marker_path, marker)

    expected_shards = set(shard_to_keys)
    actual_shards = {
        path.name for path in output_dir.iterdir() if MODEL_SHARD_RE.match(path.name)
    }
    if actual_shards != expected_shards:
        raise RuntimeError(
            f"Final shard set mismatch: missing={sorted(expected_shards - actual_shards)}, "
            f"extra={sorted(actual_shards - expected_shards)}"
        )

    marker["status"] = "complete"
    marker["completed_shards"] = sorted(completed)
    write_json_atomic(marker_path, marker)
    print(f"Merge completed: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
