#!/usr/bin/env python
"""Recall-vs-speed benchmark for pyvecdb's indexes.

Every approximate index is compared with exact search (recall@k) and, if installed,
with hnswlib (C++), using the same M and ef_construction. Examples:

    python benchmarks/run_ann.py --dataset clustered --n 10000 --dim 64
    python benchmarks/run_ann.py --dataset digits                  # real data, bundled with scikit-learn
    python benchmarks/run_ann.py --dataset npy --base base.npy --queries-file q.npy   # your own vectors
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pyvecdb.index import HNSW, BruteForce, IVFFlat


def load(args):
    rng = np.random.default_rng(args.seed)
    if args.dataset == "clustered":  # Gaussian mixture: the shape that exposes naive neighbour selection
        centers = rng.normal(0, 3, (args.clusters, args.dim))

        def pick(m):
            return centers[rng.integers(0, args.clusters, m)] + rng.normal(0, 1, (m, args.dim))

        return pick(args.n).astype(np.float32), pick(args.queries).astype(np.float32)
    if args.dataset == "uniform":
        return (rng.normal(size=(args.n, args.dim)).astype(np.float32),
                rng.normal(size=(args.queries, args.dim)).astype(np.float32))
    if args.dataset == "digits":  # real data: 8x8 handwritten digits
        from sklearn.datasets import load_digits
        X = load_digits().data.astype(np.float32)
        order = rng.permutation(len(X))
        return X[order[args.queries:]], X[order[: args.queries]]
    if args.dataset == "npy":
        return np.load(args.base).astype(np.float32), np.load(args.queries_file).astype(np.float32)
    raise SystemExit(f"unknown dataset {args.dataset}")


def recall_and_qps(search, Q, truth, k):
    search(Q[0])  # warm-up
    t0 = time.perf_counter()
    found = [search(q) for q in Q]
    qps = len(Q) / (time.perf_counter() - t0)
    rec = float(np.mean([len(set(f) & set(t)) / k for f, t in zip(found, truth)]))
    return rec, qps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="clustered", choices=["clustered", "uniform", "digits", "npy"])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--clusters", type=int, default=50)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--metric", default="cosine", choices=["cosine", "l2"])
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=100, help="ef_construction")
    ap.add_argument("--nlist", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", help="npy file of base vectors (dataset=npy)")
    ap.add_argument("--queries-file", help="npy file of query vectors (dataset=npy)")
    ap.add_argument("--out", default="benchmarks/results")
    args = ap.parse_args()

    X, Q = load(args)
    n, dim = X.shape
    print(f"dataset={args.dataset} n={n} dim={dim} queries={len(Q)} metric={args.metric} k={args.k}")

    exact = BruteForce(dim, args.metric)
    exact.add(X)
    truth = [exact.search(q, args.k)[0].tolist() for q in Q]
    results = []

    def record(name, build_s, param, value, rec, qps):
        results.append({"method": name, "build_s": round(build_s, 2), "param": param, "value": value,
                        "recall": round(rec, 4), "qps": round(qps, 1)})
        print(f"  {name:22s} {param}={value:<4} recall@{args.k}={rec:.3f}  {qps:9.0f} qps  (build {build_s:.1f}s)")

    rec, qps = recall_and_qps(lambda q: exact.search(q, args.k)[0].tolist(), Q, truth, args.k)
    record("exact (numpy)", 0.0, "-", 0, rec, qps)

    for label, heuristic in (("hnsw (heuristic)", True), ("hnsw (naive select)", False)):
        t0 = time.perf_counter()
        idx = HNSW(dim, args.metric, M=args.M, ef_construction=args.efc, heuristic=heuristic)
        idx.add(X)
        build = time.perf_counter() - t0
        for ef in (10, 20, 40, 80, 160, 320):
            r, q = recall_and_qps(lambda v, ef=ef, idx=idx: idx.search(v, args.k, ef=ef)[0].tolist(), Q, truth, args.k)
            record(label, build, "ef", ef, r, q)

    t0 = time.perf_counter()
    ivf = IVFFlat(dim, args.metric, nlist=max(2, min(args.nlist, n // 10)))
    ivf.add(X)
    build = time.perf_counter() - t0
    for nprobe in (1, 2, 4, 8, 16, 32):
        r, q = recall_and_qps(lambda v, p=nprobe: ivf.search(v, args.k, nprobe=p)[0].tolist(), Q, truth, args.k)
        record("ivf", build, "nprobe", nprobe, r, q)

    try:
        import hnswlib
        t0 = time.perf_counter()
        ref = hnswlib.Index(space="cosine" if args.metric == "cosine" else "l2", dim=dim)
        ref.init_index(max_elements=n, ef_construction=args.efc, M=args.M, random_seed=42)
        ref.add_items(X, num_threads=1)
        build = time.perf_counter() - t0
        for ef in (10, 20, 40, 80, 160, 320):
            ref.set_ef(ef)
            r, q = recall_and_qps(lambda v: ref.knn_query(v[None], args.k)[0][0].tolist(), Q, truth, args.k)
            record("hnswlib (C++)", build, "ef", ef, r, q)
    except ImportError:
        print("  (hnswlib not installed; skipping the C++ reference)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"ann_{args.dataset}_{n}x{dim}_{args.metric}"
    (out / f"{stem}.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=1))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
        style = {"hnsw (heuristic)": ("#1d5c96", "o-"), "hnsw (naive select)": ("#c2185b", "s--"),
                 "ivf": ("#2e8b57", "^-"), "hnswlib (C++)": ("#888888", "d:")}
        for name, (colour, fmt) in style.items():
            pts = [r for r in results if r["method"] == name]
            if pts:
                ax.plot([p["recall"] for p in pts], [p["qps"] for p in pts], fmt, color=colour,
                        label=name, lw=1.6, ms=5)
        ax.plot([1.0], [results[0]["qps"]], "k*", ms=11, label="exact (numpy)")
        ax.set_yscale("log")
        ax.set_xlabel(f"recall@{args.k}")
        ax.set_ylabel("queries per second (1 thread)")
        ax.set_title(f"{args.dataset}: {n:,} x {dim}, {args.metric}  (up and to the right is better)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{stem}.png")
        print(f"wrote {out / (stem + '.png')}")
    except ImportError:
        print("matplotlib not installed; skipped the plot (pip install 'pyvecdb[bench]')")


if __name__ == "__main__":
    main()
