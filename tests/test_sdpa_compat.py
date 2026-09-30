"""Full attention, not candidate truncation or multiple model decisions."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from decider.sdpa_compat import expanded_sdpa, enable_expanded_sdpa, NAME
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


@pytest.mark.parametrize("groups", [1, 4])
@pytest.mark.parametrize("masked,qlen,klen", [(False, 32, 32), (True, 32, 32), (False, 1, 48)])
def test_matches_original_attention(groups, masked, qlen, klen):
    torch.manual_seed(4)
    module = SimpleNamespace(num_key_value_groups=groups, is_causal=True)
    q = torch.randn(1, 2 * groups, qlen, 64)
    k, v = torch.randn(1, 2, klen, 64), torch.randn(1, 2, klen, 64)
    mask = torch.ones(qlen, klen, dtype=torch.bool).tril()[None, None] if masked else None
    expected, _ = sdpa_attention_forward(module, q, k, v, mask, scaling=.125)
    actual, weights = expanded_sdpa(module, q, k, v, mask, scaling=.125)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert weights is None


def test_registration_preserves_sdpa_masking_and_is_model_scoped():
    model = Mock()
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    enable_expanded_sdpa(model)
    model.set_attn_implementation.assert_called_once_with(NAME)
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original
    assert ALL_MASK_ATTENTION_FUNCTIONS[NAME] is ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]


@pytest.mark.cuda
def test_cuda_memory_efficient_kernel_matches_math_without_gqa():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.manual_seed(7)
    module = SimpleNamespace(num_key_value_groups=4, is_causal=True)
    q = torch.randn(1, 8, 256, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 256, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    with sdpa_kernel(SDPBackend.MATH):
        expected, _ = sdpa_attention_forward(module, q, k, v, None)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        actual, _ = expanded_sdpa(module, q, k, v, None)
    torch.testing.assert_close(actual.float(), expected.float(), atol=.016, rtol=.02)
