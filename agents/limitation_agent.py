from __future__ import annotations

from typing import Dict

from core.llm_client import LLMClient


class LimitationAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, context: Dict) -> Dict:
        # Debate critique path: critique each hypothesis independently.
        hypotheses = context.get("hypotheses_to_critique", [])
        if hypotheses:
            prompt = f"""
Critique each hypothesis and return JSON with one key:
- hypothesis_critiques: array of objects

Each object must contain:
- hypothesis (string)
- critique_points (array of strings)
- improvement_suggestions (array of strings)

Hypotheses:
{hypotheses}
""".strip()
            result = self.llm.generate_json(prompt)
            return {"hypothesis_critiques": result.get("hypothesis_critiques", [])}

        paper_context = context.get("paper_context", "")
        summary = context.get("paper_summary", "")
        contributions = context.get("key_contributions", [])

        prompt = f"""
Given the paper context, summary, and contributions, return JSON with:
- identified_limitations (array of strings)
- improvement_opportunities (array of strings)
- key_assumptions (array of strings)

Summary:
{summary}

Contributions:
{contributions}

Paper context:
{paper_context}
""".strip()
        result = self.llm.generate_json(prompt)
        return {
            "identified_limitations": result.get("identified_limitations", []),
            "improvement_opportunities": result.get("improvement_opportunities", []),
            "key_assumptions": result.get("key_assumptions", []),
        }
