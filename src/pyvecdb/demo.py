from __future__ import annotations

import json
from pathlib import Path

_CORPUS = Path(__file__).parent / "data" / "mini_corpus.json"


def corpus() -> dict:
    """The bundled mini corpus: documents plus labelled questions (see benchmarks/eval_rag.py)."""
    return json.loads(_CORPUS.read_text(encoding="utf-8"))


def load_demo(pipeline) -> int:
    docs = corpus()["docs"]
    for d in docs:
        pipeline.ingest(d["title"], d["text"], source="demo corpus", doc_id=d["id"])
    return len(docs)
