#!/usr/bin/env python
"""Retrieval + answering evaluation on the bundled mini corpus (or your own JSON).

Reports document-level hit@1, hit@3 and MRR for each retrieval mode and chunk size, plus
how often the pipeline answers answerable questions and abstains on unanswerable ones.

    python benchmarks/eval_rag.py
    python benchmarks/eval_rag.py --embedder ollama:nomic-embed-text --llm ollama:llama3.2

Corpus format: {"docs": [{"id", "title", "text"}], "questions": [{"q", "doc", "answerable"}]}.
The bundled corpus is tiny (12 documents, 27 questions): treat it as a regression test,
not as evidence about real-world quality. Bring your own corpus for that.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyvecdb.demo import corpus as bundled
from pyvecdb.embed import get_embedder
from pyvecdb.rag import RAGPipeline, get_generator


def evaluate(data, embedder, generator, chunk_words, overlap, mode, mmr):
    pipe = RAGPipeline(embedder, generator, chunk_words=chunk_words, chunk_overlap=overlap)
    for d in data["docs"]:
        pipe.ingest(d["title"], d["text"], doc_id=d["id"])
    answerable = [q for q in data["questions"] if q["answerable"]]
    other = [q for q in data["questions"] if not q["answerable"]]
    h1 = h3 = rr = 0.0
    answered = 0
    for q in answerable:
        hits = pipe.search(q["q"], k=5, mode=mode, mmr_lambda=mmr)
        docs = []
        for r in hits:
            if r.doc_id not in docs:
                docs.append(r.doc_id)
        rank = docs.index(q["doc"]) + 1 if q["doc"] in docs else None
        h1 += rank == 1
        h3 += rank is not None and rank <= 3
        rr += 1.0 / rank if rank else 0.0
        answered += not pipe.ask(q["q"], k=4, mode=mode, mmr_lambda=mmr)["abstained"]
    abstained = sum(pipe.ask(q["q"], k=4, mode=mode, mmr_lambda=mmr)["abstained"] for q in other)
    n = len(answerable)
    return {"chunks": pipe.stats()["chunks"], "hit@1": h1 / n, "hit@3": h3 / n, "mrr": rr / n,
            "answers_when_answerable": answered / n,
            "abstains_when_unanswerable": abstained / max(1, len(other))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedder", default="hashing")
    ap.add_argument("--llm", default="extractive")
    ap.add_argument("--corpus", help="path to a corpus JSON (default: bundled)")
    ap.add_argument("--out", default="benchmarks/results/rag_eval.json")
    args = ap.parse_args()
    data = json.loads(Path(args.corpus).read_text()) if args.corpus else bundled()
    embedder, generator = get_embedder(args.embedder), get_generator(args.llm)

    configs = [("dense", None), ("bm25", None), ("hybrid", None), ("hybrid", 0.7)]
    rows = []
    print("| chunking | retrieval | chunks | hit@1 | hit@3 | MRR | answers | abstains |")
    print("|---|---|---|---|---|---|---|---|")
    for words, overlap in ((180, 30), (40, 10)):
        for mode, mmr in configs:
            m = evaluate(data, embedder, generator, words, overlap, mode, mmr)
            label = mode + (f" + MMR {mmr}" if mmr else "")
            rows.append({"chunk_words": words, "overlap": overlap, "retrieval": label, **m})
            print(f"| {words}w/{overlap} | {label} | {m['chunks']} | {m['hit@1']:.2f} | {m['hit@3']:.2f} | "
                  f"{m['mrr']:.2f} | {m['answers_when_answerable']:.2f} | {m['abstains_when_unanswerable']:.2f} |")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"embedder": args.embedder, "llm": args.llm, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
