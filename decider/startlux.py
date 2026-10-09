"""StartLux adapter for decider.serve; upstream inference stays unmodified."""
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

from decider._startlux import jevfmt as J
from decider.systemone import certainty

MODEL = "startlux-models/StartLux-Decision-0.8B"
REVISION = "bd4f76a600e23227547fee7bfc1825e12a32764c"


def configure_environment(env):
    """Use short, user-writable Windows caches before importing the CUDA compiler."""
    if os.name == "nt":
        root = Path(tempfile.gettempdir()) / "decider"
        for name, suffix in (("TORCHINDUCTOR_CACHE_DIR", "i"), ("TRITON_CACHE_DIR", "t"),
                             ("CUDA_CACHE_PATH", "c")):
            env.setdefault(name, str(root / suffix))
        # torch 2.8's static launcher narrows a CUDA pointer to Windows' 32-bit C long.
        env["TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER"] = "0"


def is_startlux(path):
    return (Path(path) / "decision_config.json").is_file()


def row_key(row, order):
    return (row["type"], row["state"], row["instructions"],
            tuple((option["id"], option.get("criterion")) for option in row["options"]), tuple(order))


@dataclass
class Prepared:
    state: str
    questions: dict
    tokens: dict
    lengths: list


class StateTooLong(ValueError):
    pass


def prepare(tok, state, questions, max_state_tokens, group=25, keep=3):
    """Validate and bound all work before GPU admission; never truncate evidence.

    Wide choices retain upstream's multi-round algorithm. Their not-yet-selected
    finalist prompts use a conservative UTF-8 byte bound (Qwen's byte BPE cannot
    produce more tokens than bytes), including every initial option and prompt.
    This may reject a wide request near a configured limit, never undercount it.
    """
    text = J.state_text(state)
    if len(tok.encode(text, add_special_tokens=False)) > max_state_tokens:
        raise StateTooLong("complete StartLux state exceeds DECIDER_MAX_STATE_TOKENS")
    cache, lengths = {}, []
    for question in questions.values():
        if not isinstance(question, dict):
            raise ValueError("each question must be an object")
        kind = question.get("type", "choice")
        criteria = question.get("criteria", question.get("options"))
        if kind == "choice" and not isinstance(criteria, (dict, list, tuple)):
            raise ValueError("choice criteria must be a map or list of options")
        if kind == "score" and not isinstance(criteria, (dict, list, tuple)):
            raise ValueError("score criteria must contain 2..10 ordered levels")
        if kind == "choice" and isinstance(question.get("criteria"), dict) and len(criteria) == 1:
            continue  # the official implementation answers a singleton without inference
        wide = kind == "choice" and isinstance(question.get("criteria"), dict) and len(criteria) > J.MAX_OPTIONS
        specs = [question]
        if wide:
            keys = list(criteria)
            groups = -(-len(keys) // group)
            size, extra = divmod(len(keys), groups)
            specs, start, finalists = [], 0, 0
            for index in range(groups):
                end = start + size + (index < extra)
                specs.append(dict(question, criteria={key: criteria[key] for key in keys[start:end]}))
                finalists += min(keep, end-start)
                start = end
        byte_bound = 0
        for spec in specs:
            row = J.from_systemone(text, spec)
            messages, order = J.messages(row)
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok.encode(prompt, add_special_tokens=False)
            cache[row_key(row, order)] = ids
            lengths.append(len(ids))
            byte_bound += len(prompt.encode("utf-8"))
        if wide:
            while finalists > J.MAX_OPTIONS:
                groups = -(-finalists // group)
                size, extra = divmod(finalists, groups)
                lengths.extend([byte_bound] * groups)
                finalists = sum(min(keep, size + (index < extra)) for index in range(groups))
            lengths.append(byte_bound)
    return Prepared(text, questions, cache, lengths)


def configure_shapes(upstream, token_budget):
    """Keep upstream padding/capture shapes inside the deployment's GPU budget."""
    if token_budget < 128:
        raise ValueError("StartLux token budget must be at least 128 for CUDA graph capture")
    upstream.GRAPH_LENGTHS = tuple(n for n in upstream.GRAPH_LENGTHS if n <= token_budget)
    upstream.MULTI_ROW_MAX_LENGTH = min(upstream.MULTI_ROW_MAX_LENGTH, token_budget // max(upstream.GRAPH_ROWS))
    upstream.PAD_LENGTHS = tuple(n for n in upstream.PAD_LENGTHS if n < token_budget) + (token_budget,)


class StartLuxEngine:
    """All CUDA state is created and used on serve's single inference executor."""
    def __init__(self, path, device, max_state_tokens, max_row_tokens, token_budget):
        configure_environment(os.environ)
        import torch
        from decider._startlux import model as upstream
        from decider.startlux_kernels import install

        if not str(device).startswith("cuda"):
            raise ValueError("the StartLux serving adapter requires CUDA; no silent CPU fallback")
        self.dev = torch.device(device)
        torch.cuda.set_device(self.dev)
        self.dev = torch.device("cuda", torch.cuda.current_device())
        self.kernel_report, _, _ = install(upstream)
        configure_shapes(upstream, token_budget)
        self.max_row_tokens = min(max_row_tokens, token_budget)

        class CachedDecision(upstream.StartLuxDecision):
            prepared_tokens = None

            def _render(self, row, order):
                if self.prepared_tokens is not None:
                    ids = self.prepared_tokens.get(row_key(row, order))
                    if ids is not None:
                        return ids
                return super()._render(row, order)

        self.model = CachedDecision(str(path), device=device, images=False,
                                    max_length=self.max_row_tokens, max_batch_tokens=token_budget,
                                    prefill_chunk=min(upstream.PREFILL_CHUNK, token_budget))
        if not all(parameter.device == self.dev for parameter in self.model.body.parameters()):
            raise RuntimeError("StartLux model parameters are not all on the selected CUDA device")
        if not self.model.fast_kernels or not self.model.graphs:
            raise RuntimeError("StartLux requires validated fast kernels and captured CUDA graphs")
        if not (1 <= self.model.keep < self.model.group <= J.MAX_OPTIONS and 2*self.model.keep <= J.MAX_OPTIONS):
            raise ValueError("invalid StartLux wide-choice group/keep configuration")
        self.tok, self.graphs = self.model.tok, self.model.graphs
        self.max_state_tokens = max_state_tokens
        self.stats = dict(requests=0)
        questions = {"ready": {"type": "noul", "instructions": "Is this a readiness check?"}}
        difference = self.model.self_test("A readiness check.", questions)
        if difference > .02:
            raise RuntimeError(f"StartLux graph/eager probability mismatch: {difference}")
        self.kernel_report["graph_probability_difference"] = difference
        self.sealed = True

    def prepare(self, state, questions):
        return prepare(self.tok, state, questions, self.max_state_tokens,
                       self.model.group, self.model.keep)

    def decide(self, request):
        self.model.prepared_tokens = request.tokens
        try:
            answers, usage = self.model.decide(request.state, request.questions)
        finally:
            self.model.prepared_tokens = None
        for answer in answers.values():
            if "probabilities" in answer:
                answer["certainty"] = round(certainty(list(answer["probabilities"].values())), 4)
        self.stats["requests"] += 1
        return answers, usage
