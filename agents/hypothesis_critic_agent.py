from __future__ import annotations

from typing import Dict, List

from core.llm_client import LLMClient


class HypothesisCriticAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    @staticmethod
    def _clamp_score(value) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))

    def run(self, context: Dict) -> Dict:
        hypotheses: List[str] = context.get("generated_hypotheses", [])
        prompt = f"""
Critique each hypothesis and return JSON with:
- hypothesis_critique: array of objects

Each object must contain:
- hypothesis (string)
- novelty (number from 0 to 1)
- testability (number from 0 to 1)
- mechanistic_clarity (number from 0 to 1)
- feasibility (number from 0 to 1)
- improvement_suggestions (array of strings)

Hypotheses:
{hypotheses}
""".strip()

        result = self.llm.generate_json(prompt)
        raw_items = result.get("hypothesis_critique", [])
        normalized = []

        for item in raw_items if isinstance(raw_items, list) else []:
            if not isinstance(item, dict):
                continue
            suggestions = item.get("improvement_suggestions", [])
            normalized.append(
                {
                    "hypothesis": item.get("hypothesis", ""),
                    "novelty": self._clamp_score(item.get("novelty", 0.0)),
                    "testability": self._clamp_score(item.get("testability", 0.0)),
                    "mechanistic_clarity": self._clamp_score(
                        item.get("mechanistic_clarity", 0.0)
                    ),
                    "feasibility": self._clamp_score(item.get("feasibility", 0.0)),
                    "improvement_suggestions": suggestions if isinstance(suggestions, list) else [],
                }
            )

        return {"hypothesis_critique": normalized}
