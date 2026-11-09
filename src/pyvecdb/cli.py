from __future__ import annotations

import argparse
import os
import sys

from . import __version__


def _serve(args) -> int:
    import uvicorn

    from .api import create_app
    from .demo import load_demo
    from .embed import get_embedder
    from .rag import RAGPipeline, get_generator

    pipe = RAGPipeline(get_embedder(args.embedder), get_generator(args.llm), path=args.data,
                       index=args.index)
    if args.demo and not pipe.docs:
        print(f"loaded {load_demo(pipe)} demo documents")
    key = args.api_key or os.environ.get("PYVECDB_API_KEY") or None
    if args.host not in ("127.0.0.1", "localhost", "::1") and not key:
        print("warning: listening beyond localhost without --api-key; anyone who can reach this "
              "port can read and delete your documents", file=sys.stderr)
    uvicorn.run(create_app(pipe, api_key=key, cors_origins=args.cors_origin or None),
                host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pyvecdb", description="Vector search + RAG workbench")
    ap.add_argument("--version", action="version", version=f"pyvecdb {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the API and web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--data", default=None, help="snapshot file; omit for an in-memory session")
    s.add_argument("--embedder", default=os.environ.get("PYVECDB_EMBEDDER", "hashing"),
                   help="hashing | ollama:<model> | st:<model>")
    s.add_argument("--llm", default=os.environ.get("PYVECDB_LLM", "extractive"),
                   help="extractive | ollama:<model>")
    s.add_argument("--index", default="hnsw", choices=["hnsw", "ivf", "bruteforce"])
    s.add_argument("--demo", action="store_true", help="load the bundled demo documents")
    s.add_argument("--api-key", default=None, help="require this X-API-Key (or PYVECDB_API_KEY)")
    s.add_argument("--cors-origin", action="append", help="allowed origin (repeatable)")
    s.set_defaults(func=_serve)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
