from __future__ import annotations

from typing import Dict

from core.llm_client import LLMClient


class ExperimentDesignAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, context: Dict) -> Dict:
        hypotheses = context.get("generated_hypotheses", [])

        prompt = f"""
Design experiments to test the hypotheses below.
Return JSON with one key:
- proposed_experiments (array of objects)
Each object must contain:
- objective
- design
- required_resources
- evaluation_metrics
- risks
- estimated_experiment_cost_level (Low/Medium/High)
- potential_impact (integer 1-5)
- risk_factors (array of strings)

Hypotheses:
{hypotheses}
""".strip()
        result = self.llm.generate_json(prompt)
        return {"proposed_experiments": result.get("proposed_experiments", [])}
