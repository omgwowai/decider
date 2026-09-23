"""Isolated NVIDIA deployment of the official decider.serve application."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODEL = "Mapika/decider-0.8b"
REVISION = "a0a01d6f8135298f400a8c856b355793012ae971"
TORCH = "torch==2.8.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python", help="base Python for the isolated venv, or existing GPU Python with --skip-install")
    p.add_argument("--skip-install", action="store_true", help="use --python directly; never install or modify its packages")
    p.add_argument("--venv", type=Path, default=ROOT / ".venv-deploy")
    p.add_argument("--model", default=MODEL, help="Hugging Face repo ID; a nondefault model requires --revision")
    p.add_argument("--revision", help="immutable 40-character model commit; defaults to the pinned 0.8B revision")
    p.add_argument("--local-model", type=Path, help="explicit complete local model folder; no Hub download")
    p.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "huggingface")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8102)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--batch-wait-ms", type=float, default=0)
    p.add_argument("--token-budget", type=int, default=8192)
    p.add_argument("--max-pending", type=int, default=64)
    p.add_argument("--runtime", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.skip_install and not args.python:
        p.error("--skip-install requires --python pointing to a compatible GPU interpreter")
    if not 1 <= args.port <= 65535:
        p.error("--port must be in 1..65535")
    if min(args.max_batch, args.token_budget, args.max_pending) < 1 or not 0 <= args.batch_wait_ms < float("inf"):
        p.error("batch size, token budget and pending limit must be positive; wait must be finite and nonnegative")
    if not re.fullmatch(r"cuda(?::[0-9]+)?", args.device):
        p.error("this deployment requires an NVIDIA CUDA device, e.g. cuda:0; there is no CPU fallback")
    if not args.local_model:
        if args.revision is None:
            if args.model != MODEL:
                p.error("--model requires an explicit immutable --revision for a nondefault model")
            args.revision = REVISION
        if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
            p.error("--revision must be an immutable 40-character commit, not main or a tag")
    for name in ("venv", "local_model", "cache_dir"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    return args


def run(command, **kwargs):
    print("[deploy]", subprocess.list2cmdline([str(x) for x in command]), flush=True)
    subprocess.run([str(x) for x in command], check=True, **kwargs)


def prepare_python(args):
    executable = shutil.which(args.python or sys.executable)
    if not executable:
        raise RuntimeError(f"Python interpreter not found: {args.python}")
    if args.skip_install:
        return executable
    version = subprocess.check_output([executable, "-c", "import sys; print('%s.%s' % sys.version_info[:2])"], text=True).strip()
    if version not in ("3.11", "3.12"):
        raise RuntimeError(f"Managed CUDA wheels require Python 3.11–3.12, got {version}; pass --python with a supported interpreter")
    python = args.venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        run([executable, "-m", "venv", args.venv])
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", TORCH, "--index-url", TORCH_INDEX])
    if os.name == "nt":
        run([python, "-m", "pip", "install", "triton-windows>=3.4,<3.5"])
    # FLA >=0.5 does not pull the Linux-only triton distribution on Windows.
    run([python, "-m", "pip", "install", TORCH, "flash-linear-attention==0.5.2", f"{ROOT}[serve]"])
    return str(python)


def gpu_preflight(device):
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(f"torch {torch.__version__} (CUDA {torch.version.cuda}) cannot access NVIDIA CUDA")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("the selected GPU does not support the model's bfloat16 execution")
            x = torch.ones((16, 16), device=device, dtype=torch.bfloat16)
            result = (x @ x).sum().item()
            torch.cuda.synchronize(device)
            if result != 4096:
                raise RuntimeError(f"CUDA matrix multiply produced {result}, expected 4096")
        import triton
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from transformers.models.qwen3_5 import modeling_qwen3_5
        import fastapi
        import uvicorn
        print(f"[deploy] GPU preflight: {torch.cuda.get_device_name(device)}, torch={torch.__version__}, triton={triton.__version__}", flush=True)
    except Exception as e:
        raise RuntimeError(f"GPU preflight failed: {e}. Use a current NVIDIA driver and CUDA PyTorch wheels. "
                           "Windows requires matching triton-windows (torch 2.8 / Triton 3.4); --skip-install never repairs your environment.") from e


def resolve_model(args):
    if args.local_model:
        path = args.local_model
    else:
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(repo_id=args.model, revision=args.revision, cache_dir=str(args.cache_dir)))
    if not path.is_dir() or not (path / "config.json").is_file() or not (path / "decider_config.json").is_file():
        raise RuntimeError(f"Incomplete model folder: {path}; config.json and decider_config.json are required")
    if not any(path.glob("*.safetensors")):
        raise RuntimeError(f"No safetensors weights in {path}; provide a complete Decider checkpoint")
    return path.resolve()


def server_environment(args, model):
    env = os.environ.copy()
    # Do not inherit another experiment's Decider settings or an unrelated source checkout.
    for key in list(env):
        if key.startswith("DECIDER_"):
            del env[key]
    lengths = [n for n in (64, 128, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 3072, 4096, 6144, 8192)
               if n < args.token_budget] + [args.token_budget]
    env.update(PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               DECIDER_MODEL=str(model), DECIDER_DEVICE=args.device, DECIDER_WARMUP="1",
               DECIDER_MAX_BATCH=str(args.max_batch), DECIDER_BATCH_WAIT_MS=str(args.batch_wait_ms),
               DECIDER_BATCH_ADAPTIVE_WAIT_MS="0", DECIDER_GRAPH_TOKEN_BUDGET=str(args.token_budget),
               DECIDER_MAX_PENDING_REQUESTS=str(args.max_pending), DECIDER_MAX_QUEUE_ROWS=str(args.max_pending * args.max_batch),
               DECIDER_MAX_STATE_TOKENS=str(args.token_budget), DECIDER_MAX_ROW_TOKENS=str(args.token_budget),
               DECIDER_MAX_REQUEST_TOKENS=str(args.token_budget), DECIDER_REJECT_TRUNCATION="1",
               DECIDER_SHARED="0", DECIDER_SCHEMA_CACHE="0",
               DECIDER_B_BUCKETS=",".join(str(n) for n in (1, 2, 4, 8, 16, 32) if n <= args.max_batch),
               DECIDER_T_BUCKETS=",".join(map(str, lengths)))
    return env


def main(argv=None):
    args = parse_args(argv)
    if args.runtime:
        gpu_preflight(args.device)
        model = resolve_model(args)
        env = server_environment(args, model)
        print(f"[deploy] Official API: http://{args.host}:{args.port}/v1/systemone; wait for /health ok=true and cuda_ready=true", flush=True)
        run([sys.executable, "-m", "uvicorn", "decider.serve:app", "--host", args.host, "--port", args.port,
             "--workers", "1"], cwd=ROOT, env=env)
    else:
        python = prepare_python(args)
        arguments = list(sys.argv[1:] if argv is None else argv)
        # Resolve caller-relative paths before switching the working directory.
        arguments.extend(["--venv", str(args.venv), "--cache-dir", str(args.cache_dir), "--runtime"])
        if args.local_model:
            arguments.extend(["--local-model", str(args.local_model)])
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
        run([python, str(Path(__file__).resolve()), *arguments], cwd=ROOT, env=env)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"[deploy] ERROR: {e}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)
