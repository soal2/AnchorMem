import hashlib
import json
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ModelConfig(Settings):
    name: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    api_key_env: str = Field(min_length=1)
    temperature: float = Field(ge=0, le=2)
    max_output_tokens: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    retries: int = Field(ge=0)


class EmbeddingConfig(Settings):
    name: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    api_key_env: str = Field(min_length=1)
    dimensions: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    retries: int = Field(ge=0)


class ChunkConfig(Settings):
    tokenizer: str
    max_tokens: int = Field(ge=1)
    overlap_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def check_overlap(self):
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        return self


class FactConfig(Settings):
    prompt_path: str
    prior_context_tokens: int = Field(ge=0)


class RetrievalConfig(Settings):
    bm25_candidates: int = Field(ge=1)
    chunk_vector_candidates: int = Field(ge=1)
    fact_vector_candidates: int = Field(ge=1)
    bm25_weight: float = Field(ge=0)
    chunk_vector_weight: float = Field(ge=0)
    fact_vector_weight: float = Field(ge=0)
    rrf_k: int = Field(ge=1)
    question_weight: float = Field(ge=0)
    options_weight: float = Field(ge=0)
    seed_count: int = Field(ge=1)
    neighbor_count: int = Field(ge=0)
    evidence_budget_tokens: int = Field(ge=1)
    max_results: int = Field(ge=1)

    @model_validator(mode="after")
    def check_weights(self):
        if not any((self.bm25_weight, self.chunk_vector_weight, self.fact_vector_weight)):
            raise ValueError("at least one retrieval route must have positive weight")
        if self.question_weight <= 0:
            raise ValueError("question_weight must be positive")
        return self


class ServiceConfig(Settings):
    version: str
    competition_mode: bool
    database_dir: str
    snapshot_dir: str
    host: str
    port: int = Field(ge=1, le=65535)
    workers: int = Field(ge=1)
    auth_key_env: str
    log_level: str
    llm: ModelConfig
    embedding: EmbeddingConfig
    chunk: ChunkConfig
    facts: FactConfig
    retrieval: RetrievalConfig

    @model_validator(mode="after")
    def check_competition(self):
        if self.workers != 1:
            raise ValueError("SQLite per-user write locks require workers=1")
        if self.competition_mode and (self.llm.name != "gpt-4o-mini" or self.embedding.name != "text-embedding-v4"):
            raise ValueError("competition_mode requires gpt-4o-mini and text-embedding-v4")
        return self


class EvaluationConfig(Settings):
    version: str
    service_config_path: str
    dataset: str
    data_path: str
    data_sha256: str
    contract_repository_path: str
    contract_commit: str
    split: str
    seed: int
    dev_fraction: float = Field(gt=0, lt=1)
    sample_limit: int | None = Field(default=None, ge=1)
    run_label: str
    output_dir: str
    answer: ModelConfig
    judge: ModelConfig


def load_config(path: str | Path, kind: type[Settings]):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config must be a YAML mapping")
    config = kind.model_validate(raw)
    return config, path


def resolved_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / value).resolve()


def config_hash(config: Settings) -> str:
    payload = config.model_dump_json(exclude_none=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def index_fingerprint(config: ServiceConfig, prompt: str) -> str:
    data = {"embedding": [config.embedding.name, config.embedding.dimensions], "chunk": config.chunk.model_dump(), "facts": [config.llm.name, prompt]}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def write_snapshot(config: Settings, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="json")
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=True), encoding="utf-8")
    return config_hash(config)


def required_key(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable {name} is not set")
    return value
