from __future__ import annotations

import collections
import os
import re
from dataclasses import dataclass
from typing import Callable

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import download_hf_weight
from tqdm import tqdm

LayerToBank = Callable[[int, object], int | None]
DropPageCache = Callable[[str], None]


@dataclass(frozen=True)
class Nvfp4ExpertSourceSpec:
    key_pattern: re.Pattern[str]
    proj_to_role: dict[str, str]
    layer_to_bank: LayerToBank
    desc: str
    kind_map: dict[str, str] | None = None
    global_reciprocal: bool = False
    tp_shard_intermediate: bool = False



def _local_intermediate_range(
    intermediate_size: int,
    *,
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    if world_size < 1:
        raise ValueError(f"Invalid TP size: {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"Invalid TP rank {rank} for TP size {world_size}")
    if intermediate_size % world_size:
        raise ValueError(
            f"NVFP4 intermediate size {intermediate_size} "
            f"is not divisible by TP size {world_size}"
        )

    local_intermediate = intermediate_size // world_size
    if local_intermediate % 16:
        raise ValueError(
            f"NVFP4 local intermediate {local_intermediate} "
            "must be divisible by 16"
        )

    start = rank * local_intermediate
    return start, start + local_intermediate



def _shard_nvfp4_expert_tensor(
    tensor: torch.Tensor,
    *,
    role: str,
    kind: str,
    start: int,
    end: int,
) -> torch.Tensor:
    if kind == "weight_scale_2":
        return tensor

    if role in ("gate", "up"):
        if kind in ("weight", "weight_scale"):
            return tensor[start:end, :].clone()

    if role == "down":
        if kind == "weight":
            if start % 2 or end % 2:
                raise ValueError("NVFP4 weight shard must align to packed bytes")
            return tensor[:, start // 2:end // 2].clone()

        if kind == "weight_scale":
            if start % 16 or end % 16:
                raise ValueError("NVFP4 scale shard must align to 16-value groups")
            return tensor[:, start // 16:end // 16].clone()

    raise ValueError(f"Unsupported NVFP4 expert tensor: {role=}, {kind=}")


def _canon_kind(spec: "Nvfp4ExpertSourceSpec", kind: str) -> str:
    return spec.kind_map.get(kind, kind) if spec.kind_map else kind


def _ingest_global(spec: "Nvfp4ExpertSourceSpec", tensor: torch.Tensor) -> torch.Tensor:
    if spec.global_reciprocal:
        tensor = 1.0 / tensor.float()
    return tensor.to(torch.float16)


def _num_moe_layers(config) -> int:
    value = getattr(config, "num_moe_layers", None)
    if value is not None:
        return int(value)
    return int(config.num_layers) - int(getattr(config, "first_k_dense_replace", 0))


def _bank_layer(spec: Nvfp4ExpertSourceSpec, layer: int, config) -> int | None:
    bank_layer = spec.layer_to_bank(layer, config)
    if bank_layer is None:
        return None
    num_layers = _num_moe_layers(config)
    if bank_layer < 0 or bank_layer >= num_layers:
        raise ValueError(
            f"{spec.desc}: bank layer {bank_layer} for checkpoint layer {layer} "
            f"is outside [0, {num_layers})"
        )
    return bank_layer


def _kind_suffix(kind: str) -> str:
    return {"weight": "", "weight_scale": "_scale", "weight_scale_2": "_global"}[kind]


def iter_nvfp4_expert_pieces(
    model_path: str,
    config,
    spec: Nvfp4ExpertSourceSpec,
    *,
    parallel: bool = False,
    workers: int = 8,
    chunk: int = 8 << 20,
    drop_page_cache: DropPageCache | None = None,
    primary: bool = True,
):
    """One piece per routed expert: ``gate`` / ``up`` / ``down`` codes plus their ``_scale``
    (fp8 block scales) and ``_global`` (the per-tensor scale, reciprocal for quant-side dialects,
    fp16) companions, straight from the safetensors shards.

    Serial reads walk the shards in order; ``parallel`` uses the chunked O_DIRECT reader. Either
    way tensors of one expert may span shards, so they are grouped by (layer, expert) as they land.
    """
    from freetoken.models.loader import drop_page_cache as _drop
    from freetoken.models.loader import safetensors_weight_map
    from freetoken.moe.expert_pieces import per_expert_pieces

    drop = drop_page_cache or _drop
    folder = download_hf_weight(model_path)
    weight_map = safetensors_weight_map(folder)

    tp_range = None

    if spec.tp_shard_intermediate:
        tp_info = get_tp_info()
        tp_range = _local_intermediate_range(
            config.moe_intermediate_size,
            rank=tp_info.rank,
            world_size=tp_info.size,
        )

    wanted: dict[str, tuple[int, int, str]] = {}
    for name in weight_map:
        match = spec.key_pattern.match(name)
        if match is None:
            continue
        bank_layer = _bank_layer(spec, int(match.group("layer")), config)
        if bank_layer is None:
            continue
        proj = match.group("proj")
        if proj not in spec.proj_to_role:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert projection {proj!r}")
        kind = _canon_kind(spec, match.group("kind"))
        if kind not in ("weight", "weight_scale", "weight_scale_2"):
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert tensor kind {kind!r}")
        wanted[name] = (bank_layer, int(match.group("expert")), spec.proj_to_role[proj] + _kind_suffix(kind))
    expected = _num_moe_layers(config) * config.num_experts * 9
    if len(wanted) != expected:
        raise ValueError(f"{spec.desc}: found {len(wanted)} expert tensors, expected {expected}")


    def _process_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
        role_name = wanted[name][2]

        if tp_range is not None:
            if role_name.endswith("_global"):
                role = role_name.removesuffix("_global")
                kind = "weight_scale_2"
            elif role_name.endswith("_scale"):
                role = role_name.removesuffix("_scale")
                kind = "weight_scale"
            else:
                role = role_name
                kind = "weight"

            tensor = _shard_nvfp4_expert_tensor(
                tensor,
                role=role,
                kind=kind,
                start=tp_range[0],
                end=tp_range[1],
            )

        if role_name.endswith("_global"):
            tensor = _ingest_global(spec, tensor)

        return tensor

    def _serial():
        by_shard: dict[str, list[str]] = collections.defaultdict(list)
        for name, shard in weight_map.items():
            if name in wanted:
                by_shard[shard].append(name)
        for shard in tqdm(sorted(by_shard), desc=f"Loading {spec.desc}", disable=not primary):
            path = os.path.join(folder, shard)
            drop(path)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for name in by_shard[shard]:
                    tensor = f.get_tensor(name)
                    yield name, _process_tensor(name, tensor)
            drop(path)

    def _parallel():
        from freetoken.models.weight import iter_expert_tensors_parallel

        for name, tensor in iter_expert_tensors_parallel(folder, lambda n: n in wanted, workers=workers, chunk=chunk):
            yield name, _process_tensor(name, tensor)

    return per_expert_pieces(_parallel() if parallel else _serial(), wanted.get, tensors_per_expert=9)


__all__ = ["Nvfp4ExpertSourceSpec", "iter_nvfp4_expert_pieces"]
