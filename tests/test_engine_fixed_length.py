"""Fixed sequence shapes must preserve scores and reject overflow before inference."""
from types import SimpleNamespace

import pytest
import torch

from decider.engine import Engine, GRAPH_MAX_T


def cpu_engine(fixed_length):
    engine = Engine.__new__(Engine)
    engine.fixed_length = fixed_length
    engine.dev = "cpu"
    engine.tok = SimpleNamespace(pad_token_id=0)
    shapes = []

    def causal_logits(ids):
        shapes.append(tuple(ids.shape))
        values = ids.float().cumsum(dim=1)
        return torch.stack((values, -values), dim=-1)

    engine.logits_all = causal_logits
    return engine, shapes


def item(length):
    return {"ids": [1] * length, "slots": [length - 1], "nopts": [2]}


def test_fixed_padding_preserves_scores_across_variable_lengths():
    fixed, shapes = cpu_engine(192)
    bucketed, _ = cpu_engine(None)
    for length in (3, 65, 129, 192):
        expected = bucketed.score_items([item(length)])
        actual = fixed.score_items([item(length)])
        assert torch.equal(actual[0], expected[0])
    assert shapes == [(1, 192)] * 4


def test_fixed_overflow_fails_before_forward_without_truncating():
    engine, shapes = cpu_engine(64)
    with pytest.raises(ValueError, match="Input length 65 exceeds fixed_length=64"):
        engine.score_items([item(65)])
    assert shapes == []


def test_shared_prefix_path_obeys_fixed_limit():
    engine, shapes = cpu_engine(192)
    with pytest.raises(ValueError, match="exceeds fixed_length"):
        engine.score_shared([item(200), item(201)])
    assert shapes == []
    result = engine.score_shared([item(3), item(5)])
    assert len(result) == 2
    assert shapes == [(2, 192)]


@pytest.mark.parametrize("length", [0, -1, True, 1.5, GRAPH_MAX_T + 1])
def test_invalid_fixed_length_rejected_before_loading_model(length):
    with pytest.raises(ValueError, match="fixed_length must be an integer"):
        Engine("unused-model-path", fixed_length=length)
