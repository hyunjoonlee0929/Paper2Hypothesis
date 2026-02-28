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
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding
