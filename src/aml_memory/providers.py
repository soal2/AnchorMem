import json
from typing import Protocol
from urllib.parse import urlsplit

from openai import OpenAI

from .config import EmbeddingConfig, ModelConfig, required_key


class LLM(Protocol):
    def extract(self, prompt: str, messages: list[dict], chunks: list[dict]) -> list[dict]: ...
    def complete(self, system: str, user: str, json_response: bool = True) -> str: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class ProviderError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict, raw_response: str | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.raw_response = raw_response


def _provider_diagnostics(error: Exception, phase: str, model: str, response=None, api_key: str | None = None) -> dict:
    diagnostics = {"phase": phase, "model": model, "exception_type": type(error).__name__}
    if error.__cause__ is not None:
        diagnostics["cause_type"] = type(error.__cause__).__name__
        chain = []
        cause = error.__cause__
        while cause is not None:
            message = str(cause)
            if api_key:
                message = message.replace(api_key, "[REDACTED]")
            chain.append({"type": type(cause).__name__, "message": message})
            cause = cause.__cause__
        diagnostics["cause_chain"] = chain
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        diagnostics["status_code"] = status_code
    request_id = getattr(error, "request_id", None) or getattr(response, "_request_id", None)
    if request_id is not None:
        diagnostics["request_id"] = request_id
    return diagnostics


class OpenAILLM:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = OpenAI(api_key=required_key(config.api_key_env), base_url=config.base_url, timeout=config.timeout_seconds, max_retries=config.retries)

    def complete(self, system: str, user: str, json_response: bool = True) -> str:
        request = dict(
            model=self.config.name,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
        )
        if json_response:
            request["response_format"] = {"type": "json_object"}
        response = None
        try:
            response = self.client.chat.completions.create(**request)
            return response.choices[0].message.content or ""
        except Exception as error:
            raise ProviderError("Completion provider failed", _provider_diagnostics(error, "completion", self.config.name, response, self.client.api_key)) from error

    def extract(self, prompt: str, messages: list[dict], chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []
        indices = list(range(len(chunks)))
        schema = {
            "type": "object",
            "properties": {"facts": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "pattern": "\\S"},
                    "source_indices": {"type": "array", "minItems": 1,
                                       "items": {"type": "integer", "enum": indices}},
                },
                "required": ["content", "source_indices"],
                "additionalProperties": False,
            }}},
            "required": ["facts"],
            "additionalProperties": False,
        }
        body = json.dumps({
            "prior_messages": [{key: message[key] for key in ("role", "content", "timestamp")
                                if message.get(key) is not None} for message in messages],
            "new_chunks": [{"index": index, "content": chunk["content"]} for index, chunk in enumerate(chunks)],
        }, ensure_ascii=False)
        request = dict(
            model=self.config.name,
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": body}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            response_format={"type": "json_schema", "json_schema": {"name": "grounded_facts", "strict": True, "schema": schema}},
        )
        if urlsplit(self.config.base_url).hostname == "openrouter.ai":
            request["extra_body"] = {"provider": {"require_parameters": True}}
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as error:
            raise ProviderError("Extraction provider request failed", _provider_diagnostics(error, "extraction", self.config.name, api_key=self.client.api_key)) from error
        raw_response = None
        finish_reason = None
        try:
            if not response.choices:
                raise ValueError("extraction provider returned no choices")
            choice = response.choices[0]
            raw_response = choice.message.content if isinstance(choice.message.content, str) else None
            finish_reason = choice.finish_reason
            if getattr(choice.message, "refusal", None):
                raise ValueError("extraction provider refused the request")
            if finish_reason != "stop":
                raise ValueError("extraction provider did not finish normally")
            if not isinstance(raw_response, str):
                raise ValueError("extraction provider returned no text")
            result = json.loads(raw_response)
            if not isinstance(result, dict) or set(result) != {"facts"} or not isinstance(result["facts"], list):
                raise ValueError("extraction provider must return exactly a facts array")
            facts = []
            for fact in result["facts"]:
                if (not isinstance(fact, dict) or set(fact) != {"content", "source_indices"}
                        or not isinstance(fact["content"], str) or not fact["content"].strip()):
                    raise ValueError("extraction provider returned an invalid fact")
                sources = fact["source_indices"]
                if (not isinstance(sources, list) or not sources
                        or any(type(index) is not int or index < 0 or index >= len(chunks) for index in sources)):
                    raise ValueError("fact must cite at least one current chunk index")
                facts.append({"content": fact["content"], "source_chunk_ids": [chunks[index]["id"] for index in sources]})
            return facts
        except Exception as error:
            diagnostics = _provider_diagnostics(error, "extraction", self.config.name, response, self.client.api_key)
            diagnostics["validation_error"] = str(error)
            if finish_reason is not None:
                diagnostics["finish_reason"] = finish_reason
            raise ProviderError("Extraction provider returned invalid output", diagnostics, raw_response) from error


class OpenAIEmbedder:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client = OpenAI(api_key=required_key(config.api_key_env), base_url=config.base_url, timeout=config.timeout_seconds, max_retries=config.retries)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        output = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start:start + self.config.batch_size]
            response = None
            try:
                response = self.client.embeddings.create(model=self.config.name, input=batch, dimensions=self.config.dimensions, encoding_format="float")
                ordered = sorted(response.data, key=lambda item: item.index)
                if len(ordered) != len(batch):
                    raise ValueError("embedding provider returned wrong count")
                output.extend([item.embedding for item in ordered])
            except Exception as error:
                diagnostics = _provider_diagnostics(error, "embedding", self.config.name, response, self.client.api_key)
                diagnostics["batch_start"] = start
                raise ProviderError("Embedding provider failed", diagnostics) from error
        return output
