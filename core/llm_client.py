from __future__ import annotations

import json
from typing import Any, Dict

from openai import OpenAI


class LLMClient:
    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.system_prompt = (
            "You are a critical research scientist. Do not summarize passively. "
            "Identify weaknesses and propose concrete, technically plausible hypotheses. "
            "Think step-by-step internally, but output JSON only."
        )

    def generate_json(self, prompt: str, retries: int = 1) -> Dict[str, Any]:
        attempts = 0
        last_error: Exception | None = None

        while attempts <= retries:
            attempts += 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"LLM JSON generation failed after retry: {last_error}")
