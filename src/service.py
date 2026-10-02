import hmac
import os
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException

from . import __version__
from .config import ServiceConfig, config_hash
from .contracts import AddRequest, AddResponse, SearchRequest, SearchResponse
from .engine import MemoryEngine
from .providers import Embedder, LLM


def create_app(config: ServiceConfig, config_path: Path, llm: LLM, embedder: Embedder, decision_model=None, directory_selector=None) -> FastAPI:
    engine = MemoryEngine(config, config_path, llm, embedder, decision_model=decision_model, directory_selector=directory_selector)
    app = FastAPI(title="AML Memory", version=__version__)
    app.state.engine = engine
    expected_key = os.environ.get(config.auth_key_env)

    def authorize(authorization: str | None = Header(default=None), x_api_key: str | None = Header(default=None)):
        if not expected_key:
            raise HTTPException(status_code=503, detail="service authentication key is not configured")
        supplied = x_api_key or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None)
        if supplied is None or not hmac.compare_digest(supplied, expected_key):
            raise HTTPException(status_code=401, detail="invalid credentials")

    @app.get("/health")
    def health():
        model_keys = [config.llm.api_key_env, config.embedding.api_key_env]
        if config.consolidation.mode != "off":
            model_keys.append(config.consolidation.decision_model.api_key_env)
        if config.directory_selector:
            model_keys.append(config.directory_selector.api_key_env)
        return {"version": __version__, "config_hash": config_hash(config),
                "model_ready": all(bool(os.environ.get(key)) for key in model_keys), "auth_ready": bool(expected_key)}

    @app.post("/add", response_model=AddResponse, dependencies=[Depends(authorize)])
    def add(request: AddRequest):
        try:
            return engine.add(request)
        except FileExistsError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/search", response_model=SearchResponse, dependencies=[Depends(authorize)])
    def search(request: SearchRequest):
        try:
            return engine.search(request)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app
