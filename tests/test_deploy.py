"""Deployment contracts without installs, network, or CUDA execution."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("decider_deploy", Path(__file__).parents[1] / "deploy.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def checkpoint(path):
    path.mkdir()
    for name in ("config.json", "decider_config.json", "model.safetensors"):
        (path / name).write_text("test fixture", encoding="utf-8")
    return path


def test_download_uses_immutable_revision_and_local_model_never_downloads(tmp_path, monkeypatch):
    model = checkpoint(tmp_path / "model")
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(model)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    args = deploy.parse_args(["--cache-dir", str(tmp_path / "cache")])
    assert deploy.resolve_model(args) == model
    assert calls == [{"repo_id": deploy.MODEL, "revision": deploy.REVISION, "cache_dir": str(args.cache_dir)}]
    assert deploy.resolve_model(deploy.parse_args(["--local-model", str(model)])) == model
    assert len(calls) == 1


@pytest.mark.parametrize("argv", [["--revision", "main"], ["--model", "org/other"], ["--skip-install"],
                                    ["--device", "cpu"], ["--max-pending", "0"], ["--batch-wait-ms", "nan"]])
def test_unsafe_deployment_arguments_are_rejected(argv):
    with pytest.raises(SystemExit):
        deploy.parse_args(argv)


def test_skip_install_does_not_modify_existing_environment(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("skip-install must never invoke an installation subprocess")
    monkeypatch.setattr(deploy, "run", forbidden)
    monkeypatch.setattr(deploy.subprocess, "check_output", forbidden)
    args = deploy.parse_args(["--python", sys.executable, "--skip-install"])
    assert Path(deploy.prepare_python(args)).resolve() == Path(sys.executable).resolve()


def test_managed_install_uses_local_fork_and_cuda_wheels(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(deploy, "run", lambda command, **kwargs: calls.append([str(x) for x in command]))
    monkeypatch.setattr(deploy.subprocess, "check_output", lambda *args, **kwargs: "3.12\n")
    args = deploy.parse_args(["--venv", str(tmp_path / "venv")])
    deploy.prepare_python(args)
    assert any("venv" in c and str(args.venv) in c for c in calls)
    assert any(deploy.TORCH_INDEX in c and deploy.TORCH in c for c in calls)
    assert f"{deploy.ROOT}[serve]" in calls[-1]
    assert not any("decider-ai" in c for c in calls)
    if deploy.os.name == "nt":
        assert any("triton-windows>=3.4,<3.5" in c for c in calls)


def test_deployment_environment_is_strict_and_source_is_local(monkeypatch, tmp_path):
    monkeypatch.setenv("DECIDER_MODEL", "unrelated-experiment")
    monkeypatch.setenv("DECIDER_WARMUP", "0")
    monkeypatch.setenv("DECIDER_FP8", "1")
    monkeypatch.setenv("PYTHONPATH", "unrelated-checkout")
    args = deploy.parse_args([])
    env = deploy.server_environment(args, tmp_path)
    assert env["PYTHONPATH"] == str(deploy.ROOT)
    assert env["DECIDER_MODEL"] == str(tmp_path)
    assert env["DECIDER_WARMUP"] == env["DECIDER_REJECT_TRUNCATION"] == "1"
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["DECIDER_MAX_BATCH"] == "8"
    assert env["DECIDER_GRAPH_TOKEN_BUDGET"] == env["DECIDER_MAX_ROW_TOKENS"] == "8192"
    assert env["DECIDER_BATCH_ADAPTIVE_WAIT_MS"] == "0"
    assert env["DECIDER_MAX_PENDING_REQUESTS"] == "64"
    assert "DECIDER_FP8" not in env


def test_cpu_preflight_fails_instead_of_falling_back(monkeypatch):
    torch = SimpleNamespace(__version__="cpu-test", version=SimpleNamespace(cuda=None),
                            cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises(RuntimeError, match="cannot access NVIDIA CUDA"):
        deploy.gpu_preflight("cuda:0")


def test_runtime_starts_official_server_with_requested_port(tmp_path, monkeypatch):
    calls = []
    model = checkpoint(tmp_path / "model")
    monkeypatch.setattr(deploy, "gpu_preflight", lambda device: None)
    monkeypatch.setattr(deploy, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    deploy.main(["--runtime", "--local-model", str(model), "--port", "8199"])
    command, kwargs = calls[0]
    assert command[:4] == [sys.executable, "-m", "uvicorn", "decider.serve:app"]
    assert command[command.index("--port") + 1] == 8199
    assert command[command.index("--workers") + 1] == "1"
    assert kwargs["env"]["DECIDER_MODEL"] == str(model)


@pytest.mark.parametrize("budget", [512, 7000, 9000])
def test_custom_token_budget_never_pads_past_limit(budget, tmp_path):
    args = deploy.parse_args(["--token-budget", str(budget)])
    env = deploy.server_environment(args, tmp_path)
    lengths = [int(value) for value in env["DECIDER_T_BUCKETS"].split(",")]
    assert max(lengths) == budget
    assert int(env["DECIDER_MAX_ROW_TOKENS"]) == budget
    assert int(env["DECIDER_GRAPH_TOKEN_BUDGET"]) == budget
