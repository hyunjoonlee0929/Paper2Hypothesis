from __future__ import annotations

from typing import Dict

from core.llm_client import LLMClient


class HypothesisGeneratorAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, context: Dict) -> Dict:
        # Debate revision path: revise initial hypotheses using critiques.
        if context.get("hypothesis_critiques"):
            initial_hypotheses = context.get("initial_hypotheses", [])
            hypothesis_critiques = context.get("hypothesis_critiques", [])
            summary = context.get("paper_summary", "")

            revise_prompt = f"""
Revise the initial hypotheses using critiques.
Return JSON with one key:
- generated_hypotheses (array of strings, exactly 3 refined hypotheses)

Constraints:
- Keep scientific intent but improve precision and testability.
- Each revised hypothesis should include explicit mechanism hints.
- Avoid vague claims.

Paper summary:
{summary}

Initial hypotheses:
{initial_hypotheses}

Critiques:
{hypothesis_critiques}
""".strip()
            revised = self.llm.generate_json(revise_prompt)
            return {"generated_hypotheses": revised.get("generated_hypotheses", [])}

        summary = context.get("paper_summary", "")
        limitations = context.get("identified_limitations", [])
        improvements = context.get("improvement_opportunities", [])

        prompt = f"""
Generate novel and technically plausible follow-up hypotheses from this paper.
Return JSON with one key:
- generated_hypotheses (array of strings, exactly 3 items)

Summary:
{summary}

Limitations:
{limitations}

Improvement opportunities:
{improvements}
""".strip()
        result = self.llm.generate_json(prompt)
        return {"generated_hypotheses": result.get("generated_hypotheses", [])}
