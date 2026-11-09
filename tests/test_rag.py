import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from pyvecdb.demo import corpus, load_demo
from pyvecdb.embed import EmbeddingError, HashingEmbedder, OllamaEmbedder, get_embedder
from pyvecdb.rag import (
    ABSTAIN,
    BM25,
    ExtractiveGenerator,
    GenerationError,
    OllamaGenerator,
    RAGPipeline,
    build_prompt,
    chunk_text,
    extract_citations,
    mmr_select,
    rrf,
)
from pyvecdb.rag.retriever import Retrieved

TEXT = " ".join(f"Sentence number {i} talks about topic {i % 7} in some detail." for i in range(120))


# ---- chunking ------------------------------------------------------------------
def test_chunks_respect_budget_and_are_exact_slices():
    chunks = chunk_text(TEXT, max_words=50, overlap_words=12)
    assert len(chunks) > 5
    for c in chunks:
        assert c.words <= 50 and c.text == TEXT[c.start : c.end]


def test_chunks_cover_the_whole_text_and_overlap():
    chunks = chunk_text(TEXT, max_words=50, overlap_words=12)
    assert chunks[0].start == 0 and chunks[-1].end == len(TEXT.rstrip())
    for a, b in zip(chunks, chunks[1:]):
        assert b.start <= a.end  # no gap between neighbours
        assert b.start < a.end   # and they actually overlap


def test_overlong_sentence_is_split_by_words():
    long = " ".join(f"w{i}" for i in range(500)) + "."
    chunks = chunk_text(long, max_words=100, overlap_words=0)
    assert len(chunks) == 5 and all(c.words <= 100 for c in chunks)


def test_chunker_edge_cases():
    assert chunk_text("   ") == []
    assert len(chunk_text("One short sentence.")) == 1
    with pytest.raises(ValueError):
        chunk_text("x", max_words=10, overlap_words=10)


# ---- ranking pieces -------------------------------------------------------------
def test_bm25_ranks_rare_term_matches_first_and_supports_remove():
    b = BM25()
    b.add("a", "the cat sat on the mat")
    b.add("b", "error code E1234 raised by the scheduler")
    b.add("c", "scheduler scheduler scheduler for cats")
    assert b.search("E1234", 3)[0][0] == "b"
    assert b.search("cats", 3)[0][0] in {"a", "c"}
    b.remove("b")
    assert b.search("E1234", 3) == [] and len(b) == 2


def test_rrf_rewards_agreement_between_lists():
    fused = dict(rrf([["a", "b", "c"], ["c", "a", "d"]]))
    assert fused["a"] > fused["b"] and fused["c"] > fused["d"]
    assert list(fused)[:2] == ["a", "c"] or list(fused)[:2] == ["c", "a"]


def test_mmr_prefers_diverse_results_when_lambda_is_low():
    q = np.array([1.0, 0.0])
    near_dup = np.array([[1.0, 0.0], [0.999, 0.045], [0.6, 0.8]])
    near_dup = near_dup / np.linalg.norm(near_dup, axis=1, keepdims=True)
    assert mmr_select(q, near_dup, 2, lam=1.0) == [0, 1]
    assert mmr_select(q, near_dup, 2, lam=0.3) == [0, 2]


# ---- embedders ------------------------------------------------------------------
def test_hashing_embedder_is_deterministic_unit_length_and_lexically_meaningful():
    e = HashingEmbedder()
    a, b, c = e.embed_documents(["graph search layers", "searching a layered graph", "pasta with garlic"])
    assert np.allclose(np.linalg.norm(a), 1) and np.array_equal(a, e.embed_query("graph search layers"))
    assert a @ b > a @ c
    assert get_embedder("hashing:128").dim == 128
    with pytest.raises(ValueError):
        get_embedder("magic")


# ---- prompts, citations, generators ---------------------------------------------
def ctx(text, title="T", i=0):
    return Retrieved(f"d:{i}", "d", title, i, text, 1.0)


def test_prompt_is_grounded_and_numbered():
    p = build_prompt("Why?", [ctx("alpha", "A", 0), ctx("beta", "B", 1)])
    assert "[1] A\nalpha" in p and "[2] B\nbeta" in p and ABSTAIN in p and "untrusted" in p


def test_citation_extraction_flags_nonexistent_sources():
    assert extract_citations("x [1] y [3][1] z [9]", 3) == ([1, 3], [9])


def test_extractive_generator_cites_and_abstains():
    g = ExtractiveGenerator()
    out = g.generate("what does nprobe control", [ctx("Cats purr. Nprobe controls how many lists are scanned.")])
    assert "Nprobe controls" in out and out.endswith("[1]")
    assert g.generate("tungsten boiling", [ctx("Cats purr.")]) == ABSTAIN


# ---- pipeline -------------------------------------------------------------------
@pytest.fixture()
def pipe():
    p = RAGPipeline(HashingEmbedder())
    load_demo(p)
    return p


def test_ingest_is_idempotent_and_delete_removes_everything(pipe):
    n_docs, n_chunks = len(pipe.docs), len(pipe.bm25)
    d = corpus()["docs"][0]
    pipe.ingest(d["title"], d["text"], doc_id=d["id"])  # same id: replaced, not duplicated
    assert len(pipe.docs) == n_docs and len(pipe.bm25) == n_chunks
    assert pipe.delete("hnsw") and not pipe.delete("hnsw")
    assert all(r.doc_id != "hnsw" for r in pipe.search("hierarchical navigable small world", k=10))


def test_ask_cites_the_right_document_and_abstains_when_unsupported(pipe):
    out = pipe.ask("What does nprobe control in an inverted file index?")
    assert not out["abstained"] and out["citations"] and out["invalid_citations"] == []
    assert out["contexts"][0]["doc_id"] == "ivf"
    assert pipe.ask("What is the boiling point of tungsten?")["abstained"]


def test_doc_filter_and_modes(pipe):
    for mode in ("dense", "bm25", "hybrid"):
        assert pipe.search("graph layers", k=3, mode=mode)
    only = pipe.search("how many lists are scanned at query time", k=5, doc_ids=["ivf", "hnsw"])
    assert only and {r.doc_id for r in only} <= {"ivf", "hnsw"} and only[0].doc_id == "ivf"
    with pytest.raises(ValueError):
        pipe.search("x", mode="magic")


def test_mmr_search_returns_k_distinct_chunks(pipe):
    out = pipe.search("retrieval ranking", k=4, mmr_lambda=0.5)
    assert len({r.chunk_id for r in out}) == len(out) == 4


def test_validation(pipe):
    with pytest.raises(ValueError):
        pipe.ingest("t", "  ")
    with pytest.raises(ValueError):
        pipe.ingest("t", "text", doc_id="bad:id")


def test_persistence_roundtrip_and_embedder_mismatch(tmp_path):
    path = tmp_path / "db.npz"
    p = RAGPipeline(HashingEmbedder(), path=path)
    load_demo(p)
    q = "How is the fused score computed when merging ranked lists?"
    before = [r.chunk_id for r in p.search(q, k=3)]
    p2 = RAGPipeline(HashingEmbedder(), path=path)
    assert len(p2.docs) == len(p.docs) and [r.chunk_id for r in p2.search(q, k=3)] == before
    assert p2.ask(q)["contexts"]
    with pytest.raises(ValueError, match="embedder"):
        RAGPipeline(HashingEmbedder(256), path=path)


def test_map_points_and_query_projection(pipe):
    pts = pipe.map_points()
    assert len(pts) == pipe.stats()["chunks"] and all(-1.001 <= p["x"] <= 1.001 for p in pts)
    assert len(pipe.project("graph layers")) == 2
    assert RAGPipeline(HashingEmbedder()).map_points() == []


def test_retrieval_quality_on_bundled_corpus(pipe):
    qs = [q for q in corpus()["questions"] if q["answerable"]]
    hit3 = sum(any(r.doc_id == q["doc"] for r in pipe.search(q["q"], k=3)) for q in qs) / len(qs)
    assert hit3 >= 0.9, hit3
    unanswerable = [q for q in corpus()["questions"] if not q["answerable"]]
    assert all(pipe.ask(q["q"])["abstained"] for q in unanswerable)


# ---- Ollama clients against a local fake server ------------------------------------
def fake_ollama(legacy=False):
    emb = HashingEmbedder(64)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/api/embed" and not legacy:
                self._send(200, {"embeddings": emb.embed_documents(req["input"]).tolist()})
            elif self.path == "/api/embeddings" and legacy:
                self._send(200, {"embedding": emb.embed_documents([req["prompt"]])[0].tolist()})
            elif self.path == "/api/generate":
                self._send(200, {"response": " Answer from " + req["model"] + " [1] "})
            else:
                self._send(404, {"error": "not found"})

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


@pytest.mark.parametrize("legacy", [False, True])
def test_ollama_embedder_speaks_both_api_generations(legacy):
    srv, url = fake_ollama(legacy)
    try:
        e = OllamaEmbedder("some-model", host=url, batch_size=2)
        vecs = e.embed_documents(["a b c", "d e f", "g h i"])
        assert vecs.shape == (3, 64) and e.dim == 64
        assert np.allclose(np.linalg.norm(e.embed_query("a b c")), 1)
    finally:
        srv.shutdown()


def test_nomic_prefixes_are_applied():
    e = OllamaEmbedder("nomic-embed-text:latest", host="http://x")
    assert e._doc_prefix == "search_document: " and e._query_prefix == "search_query: "


def test_ollama_generator_and_failures():
    srv, url = fake_ollama()
    try:
        assert OllamaGenerator("m", host=url).generate("q", [ctx("c")]) == "Answer from m [1]"
    finally:
        srv.shutdown()
    with pytest.raises(EmbeddingError):
        OllamaEmbedder("m", host="http://127.0.0.1:9", timeout=1).embed_query("x")
    with pytest.raises(GenerationError):
        OllamaGenerator("m", host="http://127.0.0.1:9", timeout=1).generate("q", [ctx("c")])
