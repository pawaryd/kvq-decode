"""Roofline: KV-cache bytes read per decode step and the bandwidth-bound minimum time.

Only KV-cache reads are counted (weights, q and output traffic are excluded); that is
the floor an attention kernel is judged against.

Quantization overhead model (KIVI-style, asymmetric): int8/int4 store fp16 scale+zero.
  keys   per-channel, one (scale, zero) per channel per head per `group_size` tokens
  values per-token,   one (scale, zero) per token per head
fp8 is treated as a plain cast (no scales).
"""
from dataclasses import dataclass
from typing import Dict

# Peak HBM/GDDR bandwidth, GB/s (1e9 B/s)
GPUS: Dict[str, float] = {
    "T4": 320.0,
    "A100-80GB": 2039.0,
    "H100-SXM": 3350.0,
    "B200": 8000.0,
}

# (bytes per element, has scale/zero)
KV_DTYPES: Dict[str, tuple] = {
    "fp16": (2.0, False),
    "fp8": (1.0, False),
    "int8": (1.0, True),
    "int4": (0.5, True),
}

SCALE_BYTES = 2  # fp16
ZP_PER_SCALE = 2  # scale + zero


@dataclass(frozen=True)
class ModelShape:
    layers: int
    q_heads: int
    kv_heads: int
    head_dim: int


MODELS: Dict[str, ModelShape] = {
    "llama3-8b": ModelShape(32, 32, 8, 128),
    "llama3-70b": ModelShape(80, 64, 8, 128),
    "llama2-7b": ModelShape(32, 32, 32, 128),
    "qwen2.5-7b": ModelShape(28, 28, 4, 128),
}


def kv_bytes_per_token(shape: ModelShape, kv_dtype: str, group_size: int = 128) -> float:
    """KV bytes read per context token per decode step, summed over all layers."""
    elem, quant = KV_DTYPES[kv_dtype]
    per_head = 2 * shape.head_dim * elem  # K + V payload
    if quant:
        k_scales = shape.head_dim * SCALE_BYTES * ZP_PER_SCALE / group_size
        v_scales = SCALE_BYTES * ZP_PER_SCALE
        per_head += k_scales + v_scales
    return shape.layers * shape.kv_heads * per_head


def bytes_per_step(shape: ModelShape, ctx: int, batch: int, kv_dtype: str,
                   group_size: int = 128) -> float:
    return kv_bytes_per_token(shape, kv_dtype, group_size) * ctx * batch


def min_time_ms(nbytes: float, gpu: str) -> float:
    return nbytes / (GPUS[gpu] * 1e9) * 1e3
