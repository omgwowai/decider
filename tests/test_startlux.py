"""StartLux contracts using its real prompt/answer algorithms, without loading weights."""
import asyncio
import threading

import pytest

pytest.importorskip("torch")
httpx = pytest.importorskip("httpx")
from decider import serve, startlux
from decider._startlux import jevfmt as J
from decider._startlux.model import StartLuxDecision


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return "\n".join(message["content"] for message in messages) + J.THINK_OFF_SUFFIX


class Engine(startlux.StartLuxEngine):
    def __init__(self):
        self.tok = Tokenizer()
        self.max_state_tokens = 10000
        self.stats = dict(requests=0)
        self.dev, self.sealed, self.graphs = "cpu", True, {}
        self.kernel_report = dict(backend="test", validated=True)
        self.model = object.__new__(StartLuxDecision)
        self.model.group, self.model.keep, self.model.residual = 25, 3, .001
        self.model._media = self.model._kept = None
        self.lengths, self.fail = [], False
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()
        self.model._probs = self.probs

    def probs(self, rows):
        self.entered.set()
        assert self.release.wait(5), "test inference was not released"
        if self.fail:
            raise RuntimeError("inference failed")
        values, tokens = [], 0
        for row in rows:
            ids, _ = J.render_ids(row, self.tok, max_length=100000)
            self.lengths.append(len(ids))
            tokens += len(ids)
            n = len(row["options"])
            values.append([(i+1)/(n*(n+1)/2) for i in range(n)])
        return values, tokens


def test_preparation_preserves_complete_official_prompt_and_candidate_order():
    engine = Engine()
    state = {"actors": [{"name": str(i)} for i in range(10)], "last": "完整证据 at the end"}
    questions = {"pick": {"type": "choice", "instructions": "Choose", "criteria": {"z": "First", "a": "Second"}}}
    prepared = engine.prepare(state, questions)
    row = J.from_systemone(state, questions["pick"])
    ids, order = J.render_ids(row, engine.tok)
    assert prepared.tokens[startlux.row_key(row, order)] == ids
    prompt = bytes(ids).decode("utf-8")
    assert "完整证据 at the end" in prompt and '"_index": 9' in prompt
    assert prompt.index("First") < prompt.index("Second")
    assert prepared.lengths == [len(ids)]


@pytest.mark.parametrize("n", [2, 26, 27, 30, 255])
def test_wide_choice_keeps_all_probabilities_and_bounds_every_round(n):
    engine = Engine()
    questions = {"pick": {"type": "choice", "instructions": "Choose", "criteria": {str(i): "option " + str(i) for i in range(n)}}}
    direct, direct_usage = engine.model.decide("state", questions)
    engine.lengths.clear()
    prepared = engine.prepare("state", questions)
    answers, usage = engine.decide(prepared)
    assert {k: v for k, v in answers["pick"].items() if k != "certainty"} == direct["pick"]
    assert usage == direct_usage
    assert set(answers["pick"]["probabilities"]) == set(questions["pick"]["criteria"])
    assert sum(answers["pick"]["probabilities"].values()) == pytest.approx(1)
    assert len(prepared.lengths) == len(engine.lengths)
    assert all(bound >= actual for bound, actual in zip(prepared.lengths, engine.lengths))
    assert engine.model.prepared_tokens is None


def test_typed_answers_keep_official_confidence_score_and_singleton():
    engine = Engine()
    questions = {
        "flag": {"type": "noul", "instructions": "Yes?"},
        "scale": {"type": "score", "instructions": "Rate", "criteria": {"2": "high", "0": "low", "1": "medium"}},
        "only": {"type": "choice", "criteria": {"one": "Only"}},
    }
    direct, _ = engine.model.decide({}, questions)
    answers, _ = engine.decide(engine.prepare({}, questions))
    for key, value in answers.items():
        assert {k: v for k, v in value.items() if k != "certainty"} == direct[key]
    assert "certainty" not in answers["flag"]
    assert answers["only"]["certainty"] == 1


@pytest.fixture
def served(monkeypatch):
    engine = Engine()
    for key, value in dict(STARTLUX=True, MODEL_NAME=startlux.MODEL, eng=engine, gpu=None, cpu=None,
                           SCHEMA_FIRST=False, se=None, outstanding=0, pending_requests=set(),
                           MAX_REQUEST_TOKENS=100000, MAX_ROW_TOKENS=100000, MAX_ROWS=1024,
                           MAX_PENDING_REQUESTS=64, MAX_QUEUE_ROWS=4096).items():
        monkeypatch.setattr(serve, key, value)

    def run(fn):
        async def go():
            serve.start_workers()
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=serve.app, raise_app_exceptions=False), base_url="http://test") as client:
                    return await fn(client)
            finally:
                engine.release.set()
                serve._stop()
                await asyncio.sleep(0)
        return asyncio.run(go())
    return engine, run


PAYLOAD = {"state": "evidence", "questions": {"pick": {"type": "choice", "instructions": "Choose", "criteria": {"a": "First", "b": "Second"}}}}


def test_http_response_is_consumable_without_changing_probabilities(served):
    engine, run = served
    async def check(client):
        response = await client.post("/v1/systemone", json=PAYLOAD)
        assert response.status_code == 200
        answer = response.json()["answers"]["pick"]
        assert response.json()["model"] == startlux.MODEL
        assert answer["probabilities"] == {"a": 1/3, "b": 2/3}
        assert answer["confidence"] == pytest.approx(1/3)
        assert answer["certainty"] == round(startlux.certainty([1/3, 2/3]), 4)
    run(check)


@pytest.mark.parametrize("mode", ["state", "row", "total", "count", "invalid", "dependent"])
def test_rejected_requests_never_reach_inference(served, monkeypatch, mode):
    engine, run = served
    payload = PAYLOAD
    if mode == "state":
        engine.max_state_tokens = 1
    elif mode in ("row", "total", "count"):
        monkeypatch.setattr(serve, {"row": "MAX_ROW_TOKENS", "total": "MAX_REQUEST_TOKENS", "count": "MAX_ROWS"}[mode], 0)
    elif mode == "invalid":
        payload = dict(PAYLOAD, questions={"bad": None})
    else:
        payload = dict(PAYLOAD, independent=False)
    async def check(client):
        response = await client.post("/v1/systemone", json=payload)
        assert response.status_code == (422 if mode in ("invalid", "dependent") else 413)
        assert not engine.entered.is_set() and serve.outstanding == 0
    run(check)


def test_inference_failure_releases_capacity_and_next_request_succeeds(served):
    engine, run = served
    async def check(client):
        engine.fail = True
        assert (await client.post("/v1/systemone", json=PAYLOAD)).status_code == 500
        assert serve.outstanding == 0 and engine.model.prepared_tokens is None
        engine.fail = False
        assert (await client.post("/v1/systemone", json=PAYLOAD)).status_code == 200
    run(check)


def test_disconnect_does_not_free_running_inference_reservation(served, monkeypatch):
    engine, run = served
    monkeypatch.setattr(serve, "MAX_PENDING_REQUESTS", 1)
    engine.release.clear()
    async def check(client):
        first = asyncio.create_task(client.post("/v1/systemone", json=PAYLOAD))
        assert await asyncio.get_running_loop().run_in_executor(None, engine.entered.wait, 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert serve.outstanding > 0
        assert (await client.post("/v1/systemone", json=PAYLOAD)).status_code == 503
        engine.release.set()
        for _ in range(200):
            if not serve.pending_requests:
                break
            await asyncio.sleep(.005)
        assert not serve.pending_requests and serve.outstanding == 0
        assert (await client.post("/v1/systemone", json=PAYLOAD)).status_code == 200
    run(check)


def test_legacy_decide_route_keeps_its_response_schema(served):
    _, run = served
    async def check(client):
        response = await client.post("/decide", json={"context": "evidence", "schema": {
            "Which?": {"type": "choice", "options": ["First", "Second"]},
            "Yes?": {"type": "bool"},
            "Level?": {"type": "scale", "legend": ["low", "high"]}}})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["Which?"]["choice"] == "Second"
        assert body["Yes?"]["type"] == "noul"
        assert body["Level?"]["type"] == "scale"
        assert (await client.post("/decide", json={"context": "s", "schema": None})).status_code == 422
    run(check)


@pytest.mark.parametrize("budget", [128, 512, 7000, 9000, 16384])
def test_capture_and_padding_shapes_do_not_exceed_gpu_budget(budget):
    from types import SimpleNamespace
    from decider._startlux import model as upstream
    config = SimpleNamespace(GRAPH_LENGTHS=upstream.GRAPH_LENGTHS, GRAPH_ROWS=upstream.GRAPH_ROWS,
                             MULTI_ROW_MAX_LENGTH=upstream.MULTI_ROW_MAX_LENGTH, PAD_LENGTHS=upstream.PAD_LENGTHS)
    startlux.configure_shapes(config, budget)
    shapes = [(b,n) for b in config.GRAPH_ROWS for n in config.GRAPH_LENGTHS
              if b == 1 or n <= config.MULTI_ROW_MAX_LENGTH]
    assert shapes and all(b*n <= budget for b,n in shapes)
    assert max(config.PAD_LENGTHS) == budget
    assert all(next(n for n in config.PAD_LENGTHS if n >= length) <= budget
               for length in range(1, budget+1))


def test_backend_row_cap_is_checked_before_inference(served):
    engine, run = served
    engine.max_row_tokens = 1
    async def check(client):
        assert (await client.post("/v1/systemone", json=PAYLOAD)).status_code == 413
        assert not engine.entered.is_set() and serve.outstanding == 0
    run(check)
