"""Opt-in expanded KV heads for SDPA builds without a fused GQA kernel.

PyTorch's memory-efficient backend accepts equal Q/K/V head counts, but not
enable_gqa=True. Expanding just the KV heads preserves attention semantics
without materializing the quadratic attention matrix. Masking stays owned by
Transformers; cuDNN remains disabled by the existing engine policy.
"""
from types import SimpleNamespace

from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

NAME = "decider_expanded_sdpa"


def expanded_sdpa(module, query, key, value, attention_mask, **kwargs):
    groups = getattr(module, "num_key_value_groups", 1)
    key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    # The upstream SDPA adapter reads only these two module attributes. Pass
    # group=1 after expansion so it never enables the unsupported GQA kernel.
    adapter = SimpleNamespace(num_key_value_groups=1,
                              is_causal=getattr(module, "is_causal", True))
    return sdpa_attention_forward(adapter, query, key, value, attention_mask, **kwargs)


def enable_expanded_sdpa(model):
    ALL_ATTENTION_FUNCTIONS.register(NAME, expanded_sdpa)
    AttentionMaskInterface.register(NAME, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    model.set_attn_implementation(NAME)
