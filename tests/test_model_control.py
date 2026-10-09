"""HTTP model switching: drain, owner isolation, rollback and local-only admission."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip('fastapi')
httpx = pytest.importorskip('httpx')
from decider import serve, model_catalog
from test_serve_http import FakeEngine, QUESTIONS


@pytest.fixture
def control(monkeypatch):
    rows = [dict(id=x, name=x, available=True) for x in ('old', 'new')]
    catalog = SimpleNamespace(active='old', refresh=lambda: rows,
                              resolve=lambda key: {'old': 'old', 'new': 'new'}[key])
    events = []
    for name, value in dict(MODEL='old', MODEL_NAME='old', eng=FakeEngine(), gpu=None, cpu=None,
                            batcher_task=None, schema_task=None, STARTLUX=False, SCHEMA_FIRST=False,
                            pending_requests=set(), outstanding=0, model_catalog=catalog,
                            model_switch_task=None, model_switch=dict(state='idle', error=None)).items():
        monkeypatch.setattr(serve, name, value)

    def unload():
        events.append(('unload', threading.get_ident()))
        serve.eng = None

    def load():
        events.append((serve.MODEL, threading.get_ident()))
        serve.eng = FakeEngine()
        serve.MODEL_NAME = serve.MODEL

    monkeypatch.setattr(serve, '_unload_engine', unload)
    monkeypatch.setattr(serve, '_load_engine', load)

    async def go(fn):
        serve.start_workers()
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=serve.app),
                                         base_url='http://127.0.0.1:8104') as client:
                await fn(client)
        finally:
            serve._stop()
            await asyncio.sleep(0)
    return lambda fn: asyncio.run(go(fn)), events, load


def test_switch_drains_disconnected_request_and_owns_gpu(control):
    run, events, _ = control
    entered, release = threading.Event(), threading.Event()
    original = serve.eng.score_items

    def blocking(items, temperature=1):
        entered.set()
        assert release.wait(3)
        return original(items, temperature)
    serve.eng.score_items = blocking

    async def go(client):
        old = asyncio.create_task(client.post('/v1/systemone', json={'state': 's', 'questions': QUESTIONS}))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            old.cancel()
            with pytest.raises(asyncio.CancelledError):
                await old
            response = await client.post('/v1/model-control', json={'model': 'new'})
            assert response.status_code == 202
            assert (await client.get('/v1/model-control')).json()['pending']
            assert not (await client.get('/health')).json()['ok']
            assert (await client.post('/v1/model-control', json={'model': 'old'})).status_code == 409
            assert (await client.post('/v1/systemone', json={'state': 's', 'questions': QUESTIONS})).status_code == 503
            assert not events
        finally:
            release.set()
        await serve.model_switch_task
        status = (await client.get('/v1/model-control')).json()
        assert status['active'] == 'new' and status['ready'] and not status['pending']
        assert (await client.post('/v1/systemone', json={'state': 's', 'questions': QUESTIONS})).json()['model'] == 'new'
        assert serve.outstanding == 0
        assert [event for event, _ in events] == ['unload', 'new']
        assert len({owner for _, owner in events}) == 1
        assert events[0][1] != threading.get_ident()
    run(go)


@pytest.mark.parametrize('restore_fails', [False, True])
def test_failed_load_never_claims_target_active(control, monkeypatch, restore_fails):
    run, _, original = control

    def load():
        if serve.MODEL == 'new' or restore_fails:
            raise RuntimeError('fixture load failure')
        original()
    monkeypatch.setattr(serve, '_load_engine', load)

    async def go(client):
        assert (await client.post('/v1/model-control', json={'model': 'new'})).status_code == 202
        await serve.model_switch_task
        status = (await client.get('/v1/model-control')).json()
        assert status['switch']['state'] == 'failed'
        assert 'fixture load failure' in status['switch']['error']
        assert status['ready'] is not restore_fails
        assert status['active'] == (None if restore_fails else 'old')
        response = await client.post('/v1/systemone', json={'state': 's', 'questions': QUESTIONS})
        assert response.status_code == (503 if restore_fails else 200)
    run(go)


@pytest.mark.parametrize('headers', [{'Origin': 'https://evil.example'}, {'Host': 'evil.example'}])
def test_switch_rejects_cross_origin(control, headers):
    run, events, _ = control
    async def go(client):
        assert (await client.post('/v1/model-control', json={'model': 'new'}, headers=headers)).status_code == 403
        assert not events
    run(go)


def test_current_model_is_noop_and_unknown_fields_rejected(control):
    run, events, _ = control
    async def go(client):
        response = await client.post('/v1/model-control', json={'model': 'old'})
        assert response.json()['active'] == 'old'
        assert not events
        assert (await client.post('/v1/model-control', json={'model': 'new', 'path': '/tmp/model'})).status_code == 422
    run(go)


def test_catalog_pins_cached_models_and_preserves_custom_startup(monkeypatch, tmp_path):
    def cached(preset):
        if preset['id'].startswith('Mapika/'):
            return str(tmp_path / 'mapika')
        raise FileNotFoundError('not cached')
    monkeypatch.setattr(model_catalog, 'cached_path', cached)
    catalog = model_catalog.Catalog(str(tmp_path / 'custom'))
    assert catalog.active == 'startup'
    assert catalog.resolve('startup') == str(tmp_path / 'custom')
    assert catalog.resolve('Mapika/decider-0.8b') == str(tmp_path / 'mapika')
    assert not next(row for row in catalog.rows if row['id'] == model_catalog.MODEL)['available']
    with pytest.raises(ValueError):
        catalog.resolve(model_catalog.MODEL)
    with pytest.raises(ValueError):
        catalog.resolve('../../arbitrary')
