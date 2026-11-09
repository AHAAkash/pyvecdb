import pytest
from fastapi.testclient import TestClient

from pyvecdb.api import create_app
from pyvecdb.demo import load_demo
from pyvecdb.embed import EmbeddingError, HashingEmbedder
from pyvecdb.rag import RAGPipeline

KEY = {"X-API-Key": "s3cret"}


@pytest.fixture()
def client():
    return TestClient(create_app(RAGPipeline(HashingEmbedder()), api_key="s3cret"))


def test_health_is_open_but_everything_else_needs_the_key(client):
    assert client.get("/api/health").json()["auth"] is True
    for method, path in [("get", "/api/stats"), ("get", "/api/docs"), ("get", "/api/map"),
                         ("post", "/api/demo"), ("delete", "/api/docs/x")]:
        assert getattr(client, method)(path).status_code == 401
        assert getattr(client, method)(path, headers={"X-API-Key": "wrong"}).status_code == 401


def test_full_flow(client):
    assert client.post("/api/demo", headers=KEY).json()["loaded"] == 12
    docs = client.get("/api/docs", headers=KEY).json()["documents"]
    assert len(docs) == 12
    r = client.post("/api/ask", json={"query": "What does nprobe control?"}, headers=KEY).json()
    assert r["contexts"][0]["doc_id"] == "ivf" and r["citations"] and len(r["query_xy"]) == 2
    s = client.post("/api/search", json={"query": "keyword ranking", "mode": "bm25", "k": 3}, headers=KEY).json()
    assert s["results"] and s["mode"] == "bm25"
    assert len(client.get("/api/map", headers=KEY).json()["points"]) >= 12
    assert client.delete("/api/docs/ivf", headers=KEY).status_code == 200
    assert client.delete("/api/docs/ivf", headers=KEY).status_code == 404


def test_validation_errors(client):
    assert client.post("/api/docs", json={"title": "", "text": "x"}, headers=KEY).status_code == 422
    assert client.post("/api/docs", json={"title": "t", "text": "   "}, headers=KEY).status_code == 422
    assert client.post("/api/search", json={"query": "x", "mode": "magic"}, headers=KEY).status_code == 422
    assert client.post("/api/search", json={"query": "x", "k": 999}, headers=KEY).status_code == 422


def test_security_headers_and_no_wildcard_cors(client):
    r = client.get("/", headers={"Origin": "https://evil.example"})
    assert r.status_code == 200 and "script-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
    assert "access-control-allow-origin" not in r.headers
    assert "<script src=\"/static/app.js\">" in r.text


def test_only_explicit_cors_origins_are_allowed():
    app = create_app(RAGPipeline(HashingEmbedder()), api_key="", cors_origins=["https://ok.example"])
    c = TestClient(app)
    ok = c.get("/api/health", headers={"Origin": "https://ok.example"})
    bad = c.get("/api/health", headers={"Origin": "https://evil.example"})
    assert ok.headers["access-control-allow-origin"] == "https://ok.example"
    assert "access-control-allow-origin" not in bad.headers


def test_hostile_titles_round_trip_as_inert_data(client):
    title = "<img src=x onerror=alert(1)>"
    client.post("/api/docs", json={"title": title, "text": "Some text here."}, headers=KEY)
    assert client.get("/api/docs", headers=KEY).json()["documents"][0]["title"] == title


def test_backend_failure_becomes_502():
    class Down(HashingEmbedder):
        def embed_documents(self, texts):
            raise EmbeddingError("model server is down")

    c = TestClient(create_app(RAGPipeline(Down()), api_key=""))
    r = c.post("/api/docs", json={"title": "t", "text": "some text"})
    assert r.status_code == 502 and "model server is down" in r.json()["detail"]


def test_open_server_when_no_key_configured():
    c = TestClient(create_app(RAGPipeline(HashingEmbedder()), api_key=""))
    assert c.get("/api/stats").status_code == 200
    assert c.get("/api/health").json()["read_only"] is False


def test_read_only_app_allows_queries_but_blocks_mutations():
    app = create_app(RAGPipeline(HashingEmbedder()), api_key="", read_only=True)
    load_demo(app.state.pipeline)
    c = TestClient(app)

    assert c.get("/api/health").json()["read_only"] is True
    assert len(c.get("/api/docs").json()["documents"]) == 12
    assert c.post("/api/search", json={"query": "nprobe"}).status_code == 200
    assert c.post("/api/docs", json={"title": "new", "text": "new content"}).status_code == 403
    assert c.delete("/api/docs/ivf").status_code == 403
    assert c.post("/api/demo").status_code == 403
