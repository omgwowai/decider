"""Validated CUDA bindings for StartLux on the existing PyTorch/FLA stack.

No causal-conv1d installation or accelerated-path flag spoofing is required.
"""
import inspect
import torch
import transformers
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from decider.engine import fused_causal_conv1d_fn, set_attention_backend_policy


def load_model(path, device):
    config = AutoConfig.from_pretrained(path)
    if config.model_type != 'qwen3_5':
        raise ValueError('Windows adapter supports the pinned Qwen3.5 StartLux model')
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        path, dtype=torch.bfloat16).to(device)
    return model, getattr(model.model, 'language_model', model.model)


def install(startlux_model):
    if not torch.cuda.is_available():
        raise RuntimeError('Windows StartLux requires an actual CUDA device')
    originals = {name: inspect.unwrap(getattr(mq, name)) for name in (
        'causal_conv1d_fn', 'causal_conv1d_update',
        'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule')}
    set_attention_backend_policy()
    bindings = dict(
        causal_conv1d_fn=torch.compile(fused_causal_conv1d_fn, fullgraph=True, dynamic=True),
        causal_conv1d_update=torch.compile(originals['causal_conv1d_update'], fullgraph=True, dynamic=True),
        torch_chunk_gated_delta_rule=chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule=fused_recurrent_gated_delta_rule)
    report = dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
        backend='windows-fla-triton-and-torch-inductor', checks=[])

    def compare(label, actual, expected, atol=.02):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=.02)
        report['checks'].append(dict(name=label, device=str(actual.device),
            max_absolute_error=(actual.float()-expected.float()).abs().max().item()))

    with torch.inference_mode(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(17)
        q, k, v = [torch.randn(1, 128, 16, 128, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
        g = -torch.rand(1, 128, 16, device='cuda') * .1
        beta = torch.rand(1, 128, 16, device='cuda', dtype=torch.bfloat16)
        expected, expected_state = originals['torch_chunk_gated_delta_rule'](
            q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
        actual, state = chunk_gated_delta_rule(q, k, v, g, beta,
            output_final_state=True, use_qk_l2norm_in_kernel=True)
        compare('chunk_output', actual, expected)
        compare('chunk_state', state, expected_state)
        args = [tensor[:, :1].contiguous() for tensor in (q, k, v)]
        options = dict(g=g[:, :1].contiguous(), beta=beta[:, :1].contiguous(),
            initial_state=expected_state, output_final_state=True, use_qk_l2norm_in_kernel=True)
        expected, expected_state_next = originals['torch_recurrent_gated_delta_rule'](*args, **options)
        actual, state_next = fused_recurrent_gated_delta_rule(*args, **options)
        compare('recurrent_output', actual, expected)
        compare('recurrent_state', state_next, expected_state_next)
        hidden = torch.randn(1, 6144, 128, device='cuda', dtype=torch.bfloat16)
        weight = torch.randn(6144, 4, device='cuda', dtype=torch.bfloat16)
        compare('compiled_conv', bindings['causal_conv1d_fn'](hidden, weight, activation='silu'),
            originals['causal_conv1d_fn'](hidden, weight, activation='silu'), atol=.125)
        conv_state = hidden[:, :, -4:].clone()
        reference_state = conv_state.clone()
        compare('compiled_conv_update', bindings['causal_conv1d_update'](hidden[:, :, :1].contiguous(),
            conv_state, weight, activation='silu'), originals['causal_conv1d_update'](
            hidden[:, :, :1].contiguous(), reference_state, weight, activation='silu'), atol=.125)
        compare('conv_state', conv_state, reference_state, atol=0)
        torch.cuda.synchronize()
    # Publish only after actual CUDA execution and numerical comparisons succeed.
    for name, function in bindings.items():
        setattr(mq, name, function)

    def fast_kernels_active(path):
        return (AutoConfig.from_pretrained(path).model_type == 'qwen3_5'
            and torch.cuda.is_available()
            and all(getattr(mq, name) is function for name, function in bindings.items()))

    startlux_model.fast_kernels_active = fast_kernels_active
    startlux_model.load_model = load_model
    report['validated'] = True
    return report, originals, bindings
