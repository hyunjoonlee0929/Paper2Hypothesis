from __future__ import annotations

from typing import Dict, List

from core.llm_client import LLMClient


class SummaryAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, context: Dict) -> Dict:
        paper_context = context.get("paper_context", "")
        prompt = f"""
Analyze the following paper context and return JSON with keys:
- paper_summary (string)
- key_contributions (array of strings)

Paper context:
{paper_context}
""".strip()
        result = self.llm.generate_json(prompt)

        return {
            "paper_summary": result.get("paper_summary", ""),
            "key_contributions": result.get("key_contributions", []),
        }
