"""FastAPI application: JSON API under /api, single-page UI at /."""
from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .demo import load_demo
from .embed import BackendError, get_embedder
from .rag import RAGPipeline, get_generator

STATIC = Path(__file__).parent / "static"
_CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
        "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


class DocIn(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=1_000_000)
    source: str | None = Field(default=None, max_length=500)


class QueryIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50)
    mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    mmr: float | None = Field(default=None, gt=0, le=1, description="MMR lambda; 1 = no diversity")


def create_app(pipeline: RAGPipeline | None = None, api_key: str | None = None,
               cors_origins: list[str] | None = None,
               read_only: bool | None = None) -> FastAPI:
    """Build the app. Defaults come from PYVECDB_EMBEDDER, PYVECDB_LLM, PYVECDB_DATA and
    PYVECDB_API_KEY so ``uvicorn --factory pyvecdb.api:create_app`` works.
    Set read_only=True or PYVECDB_READ_ONLY=1 to disable document mutations."""
    if pipeline is None:
        pipeline = RAGPipeline(get_embedder(os.environ.get("PYVECDB_EMBEDDER", "hashing")),
                               get_generator(os.environ.get("PYVECDB_LLM", "extractive")),
                               path=os.environ.get("PYVECDB_DATA") or None)
    key = api_key if api_key is not None else (os.environ.get("PYVECDB_API_KEY") or None)
    read_only = read_only if read_only is not None else os.environ.get("PYVECDB_READ_ONLY") == "1"

    app = FastAPI(title="pyvecdb", version=__version__,
                  docs_url="/api/swagger", openapi_url="/api/openapi.json", redoc_url=None)
    app.state.pipeline = pipeline
    if cors_origins:  # same-origin by default; opt in explicitly, never "*"
        app.add_middleware(CORSMiddleware, allow_origins=cors_origins,
                           allow_methods=["GET", "POST", "DELETE"],
                           allow_headers=["Content-Type", "X-API-Key"])

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        if not request.url.path.startswith(("/api/swagger", "/api/openapi")):
            resp.headers.setdefault("Content-Security-Policy", _CSP)
        return resp

    @app.exception_handler(BackendError)
    async def backend_error(_: Request, exc: BackendError):
        return JSONResponse({"detail": str(exc)}, status_code=502)

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if key and not (x_api_key and secrets.compare_digest(x_api_key.encode(), key.encode())):
            raise HTTPException(401, "missing or invalid X-API-Key header")

    def require_writable() -> None:
        if read_only:
            raise HTTPException(403, "this instance is read-only")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__, "auth": bool(key)}

    api = APIRouter(prefix="/api", dependencies=[Depends(require_key)])

    @api.get("/stats")
    def stats():
        return pipeline.stats()

    @api.get("/docs")
    def list_docs():
        return {"documents": pipeline.list_docs()}

    @api.post("/docs", status_code=201, dependencies=[Depends(require_writable)])
    def add_doc(doc: DocIn):
        try:
            return pipeline.ingest(doc.title, doc.text, doc.source)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @api.delete("/docs/{doc_id}", dependencies=[Depends(require_writable)])
    def delete_doc(doc_id: str):
        if not pipeline.delete(doc_id):
            raise HTTPException(404, "no such document")
        return {"deleted": doc_id}

    @api.post("/search")
    def search(q: QueryIn):
        hits = pipeline.search(q.query, k=q.k, mode=q.mode, mmr_lambda=q.mmr)
        return {"results": [h.to_dict() for h in hits], "mode": q.mode,
                "query_xy": pipeline.project(q.query)}

    @api.post("/ask")
    def ask(q: QueryIn):
        out = pipeline.ask(q.query, k=min(q.k, 12), mode=q.mode, mmr_lambda=q.mmr)
        out["query_xy"] = pipeline.project(q.query)
        return out

    @api.post("/demo", dependencies=[Depends(require_writable)])
    def demo():
        return {"loaded": load_demo(pipeline)}

    @api.get("/map")
    def embedding_map():
        return {"points": pipeline.map_points()}

    app.include_router(api)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    return app
