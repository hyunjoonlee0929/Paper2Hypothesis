from __future__ import annotations

from typing import List

from openai import OpenAI


class OpenAIEmbedder:
    def __init__(self, client: OpenAI, model: str = "text-embedding-3-small") -> None:
        self.client = client
        self.model = model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
        except Exception as exc:
            msg = str(exc).lower()
            if "insufficient_quota" in msg:
                raise RuntimeError(
                    "OPENAI_INSUFFICIENT_QUOTA: Your OpenAI account has no remaining quota."
                ) from exc
            if "invalid_api_key" in msg or "incorrect api key" in msg:
                raise RuntimeError(
                    "OPENAI_INVALID_API_KEY: Provided OpenAI API key is invalid."
                ) from exc
            if "rate_limit" in msg:
                raise RuntimeError(
                    "OPENAI_RATE_LIMIT: OpenAI rate limit exceeded. Retry later."
                ) from exc
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(model=self.model, input=[text])
        except Exception as exc:
            msg = str(exc).lower()
            if "insufficient_quota" in msg:
                raise RuntimeError(
                    "OPENAI_INSUFFICIENT_QUOTA: Your OpenAI account has no remaining quota."
                ) from exc
            if "invalid_api_key" in msg or "incorrect api key" in msg:
                raise RuntimeError(
                    "OPENAI_INVALID_API_KEY: Provided OpenAI API key is invalid."
                ) from exc
            if "rate_limit" in msg:
                raise RuntimeError(
                    "OPENAI_RATE_LIMIT: OpenAI rate limit exceeded. Retry later."
                ) from exc
            raise RuntimeError(f"Query embedding generation failed: {exc}") from exc
        return response.data[0].embedding
