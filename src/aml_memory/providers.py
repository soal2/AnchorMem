import json
from typing import Protocol

from openai import OpenAI

from .config import EmbeddingConfig, ModelConfig, required_key


class LLM(Protocol):
    def extract(self, prompt: str, messages: list[dict], chunks: list[dict]) -> list[dict]: ...
    def complete(self, system: str, user: str, json_response: bool = True) -> str: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


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
        response = self.client.chat.completions.create(**request)
        return response.choices[0].message.content or ""

    def extract(self, prompt: str, messages: list[dict], chunks: list[dict]) -> list[dict]:
        body = json.dumps({"prior_messages": messages, "new_chunks": chunks}, ensure_ascii=False)
        result = json.loads(self.complete(prompt, body))
        if not isinstance(result, dict) or not isinstance(result.get("facts"), list):
            raise ValueError("fact model must return a JSON object with facts array")
        return result["facts"]


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
            response = self.client.embeddings.create(model=self.config.name, input=batch, dimensions=self.config.dimensions)
            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise ValueError("embedding provider returned wrong count")
            output.extend([item.embedding for item in ordered])
        return output
