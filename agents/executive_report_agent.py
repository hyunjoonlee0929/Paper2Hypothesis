from __future__ import annotations

from typing import Dict, List

from agents.experiment_design_agent import ExperimentDesignAgent
from agents.hypothesis_critic_agent import HypothesisCriticAgent
from agents.hypothesis_generator_agent import HypothesisGeneratorAgent
from agents.limitation_agent import LimitationAgent
from agents.novelty_check_agent import NoveltyCheckAgent
from agents.summary_agent import SummaryAgent
from core.embedding import OpenAIEmbedder
from core.vector_store import FAISSVectorStore


class ExecutiveReportAgent:
    def __init__(
        self,
        summary_agent: SummaryAgent,
        limitation_agent: LimitationAgent,
        hypothesis_agent: HypothesisGeneratorAgent,
        hypothesis_critic_agent: HypothesisCriticAgent,
        experiment_agent: ExperimentDesignAgent,
        novelty_agent: NoveltyCheckAgent,
        embedder: OpenAIEmbedder,
        vector_store: FAISSVectorStore,
    ) -> None:
        self.summary_agent = summary_agent
        self.limitation_agent = limitation_agent
        self.hypothesis_agent = hypothesis_agent
        self.hypothesis_critic_agent = hypothesis_critic_agent
        self.experiment_agent = experiment_agent
        self.novelty_agent = novelty_agent
        self.embedder = embedder
        self.vector_store = vector_store

    def _retrieve_context(self, chunks: List[str]) -> str:
        embeddings = self.embedder.embed_texts(chunks)
        self.vector_store.build(embeddings, chunks)

        query_text = "core methodology limitations assumptions contribution findings"
        query_embedding = self.embedder.embed_query(query_text)
        retrieved = self.vector_store.retrieve(query_embedding, top_k=5, query_text=query_text)
        return "\n\n".join(item.text for item in retrieved)

    @staticmethod
    def _validate_schema(report: Dict) -> Dict:
        allowed_cost_levels = {"Low", "Medium", "High"}

        def clamp01(value) -> float:
            if not isinstance(value, (int, float)):
                return 0.0
            return max(0.0, min(1.0, float(value)))

        def clamp15(value) -> int:
            if not isinstance(value, (int, float)):
                return 1
            return max(1, min(5, int(round(value))))

        def to_string_list(value) -> List[str]:
            if not isinstance(value, list):
                return []
            return [str(item) for item in value if isinstance(item, (str, int, float))]

        def normalize_cost_level(value) -> str:
            if isinstance(value, str) and value in allowed_cost_levels:
                return value
            return "Medium"

        raw_critiques = report.get("hypothesis_critique", [])
        critiques = []
        for item in raw_critiques if isinstance(raw_critiques, list) else []:
            if not isinstance(item, dict):
                continue
            suggestions = item.get("improvement_suggestions", [])
            critiques.append(
                {
                    "hypothesis": item.get("hypothesis", ""),
                    "novelty": clamp01(item.get("novelty", 0.0)),
                    "testability": clamp01(item.get("testability", 0.0)),
                    "mechanistic_clarity": clamp01(item.get("mechanistic_clarity", 0.0)),
                    "feasibility": clamp01(item.get("feasibility", 0.0)),
                    "improvement_suggestions": to_string_list(suggestions),
                }
            )

        raw_experiments = report.get("proposed_experiments", [])
        experiments = []
        for item in raw_experiments if isinstance(raw_experiments, list) else []:
            if not isinstance(item, dict):
                continue
            experiments.append(
                {
                    "objective": item.get("objective", ""),
                    "design": item.get("design", ""),
                    "required_resources": item.get("required_resources", ""),
                    "evaluation_metrics": item.get("evaluation_metrics", ""),
                    "risks": item.get("risks", ""),
                    "estimated_experiment_cost_level": normalize_cost_level(
                        item.get("estimated_experiment_cost_level", "Medium")
                    ),
                    "potential_impact": clamp15(item.get("potential_impact", 3)),
                    "risk_factors": to_string_list(item.get("risk_factors", [])),
                }
            )

        novelty = report.get("novelty_analysis", {})
        raw_papers = novelty.get("recommended_papers", []) if isinstance(novelty, dict) else []
        normalized_papers = []
        for paper in raw_papers if isinstance(raw_papers, list) else []:
            if not isinstance(paper, dict):
                continue
            normalized_papers.append(
                {
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", ""),
                    "year": paper.get("year", ""),
                    "abstract": paper.get("abstract", ""),
                    "semantic_similarity": clamp01(paper.get("semantic_similarity", 0.0)),
                    "is_high_similarity": bool(paper.get("is_high_similarity", False)),
                    "matched_hypothesis": paper.get("matched_hypothesis", ""),
                }
            )

        raw_query_plans = novelty.get("query_plans", []) if isinstance(novelty, dict) else []
        query_plans = []
        for plan in raw_query_plans if isinstance(raw_query_plans, list) else []:
            if not isinstance(plan, dict):
                continue
            triple = plan.get("keyword_triple", {})
            if not isinstance(triple, dict):
                triple = {}
            query_plans.append(
                {
                    "hypothesis": plan.get("hypothesis", ""),
                    "keyword_triple": {
                        "subject": triple.get("subject", ""),
                        "relation": triple.get("relation", ""),
                        "mechanism": triple.get("mechanism", ""),
                    },
                    "exact_phrase_query": plan.get("exact_phrase_query", ""),
                    "keyword_and_query": plan.get("keyword_and_query", ""),
                    "mechanism_focused_query": plan.get("mechanism_focused_query", ""),
                }
            )

        key_assumptions = to_string_list(report.get("key_assumptions", []))
        required_resources = to_string_list(report.get("required_resources", []))
        if not required_resources:
            for experiment in experiments:
                resource_text = str(experiment.get("required_resources", "")).strip()
                if resource_text:
                    required_resources.append(resource_text)

        if report.get("estimated_experiment_cost_level") in allowed_cost_levels:
            estimated_experiment_cost_level = report.get("estimated_experiment_cost_level")
        else:
            level_to_score = {"Low": 1, "Medium": 2, "High": 3}
            score_to_level = {1: "Low", 2: "Medium", 3: "High"}
            inferred = 2
            for experiment in experiments:
                inferred = max(
                    inferred,
                    level_to_score.get(
                        experiment.get("estimated_experiment_cost_level", "Medium"), 2
                    ),
                )
            estimated_experiment_cost_level = score_to_level[inferred]

        if isinstance(report.get("potential_impact"), (int, float)):
            potential_impact = clamp15(report.get("potential_impact"))
        else:
            impacts = [exp.get("potential_impact", 3) for exp in experiments]
            potential_impact = clamp15(max(impacts) if impacts else 3)

        risk_factors = to_string_list(report.get("risk_factors", []))
        if not risk_factors:
            for experiment in experiments:
                risk_factors.extend(to_string_list(experiment.get("risk_factors", [])))
                risks_text = str(experiment.get("risks", "")).strip()
                if risks_text:
                    risk_factors.append(risks_text)

        # Deduplicate while preserving order.
        dedup_required_resources = list(dict.fromkeys(required_resources))
        dedup_risk_factors = list(dict.fromkeys(risk_factors))
        generated_hypotheses = to_string_list(report.get("generated_hypotheses", []))
        key_contributions = to_string_list(report.get("key_contributions", []))
        identified_limitations = to_string_list(report.get("identified_limitations", []))
        improvement_opportunities = to_string_list(report.get("improvement_opportunities", []))

        output = {
            "paper_summary": report.get("paper_summary", ""),
            "key_contributions": key_contributions,
            "key_assumptions": key_assumptions,
            "identified_limitations": identified_limitations,
            "improvement_opportunities": improvement_opportunities,
            "generated_hypotheses": generated_hypotheses,
            "required_resources": dedup_required_resources,
            "estimated_experiment_cost_level": estimated_experiment_cost_level,
            "potential_impact": potential_impact,
            "risk_factors": dedup_risk_factors,
            "hypothesis_critique": critiques,
            "proposed_experiments": experiments,
            "novelty_analysis": {
                "similar_papers_found": bool(novelty.get("similar_papers_found", False))
                if isinstance(novelty, dict)
                else False,
                "recommended_papers": normalized_papers,
                "novelty_score": clamp01(novelty.get("novelty_score", 0.0))
                if isinstance(novelty, dict)
                else 0.0,
                "high_similarity_found": bool(novelty.get("high_similarity_found", False))
                if isinstance(novelty, dict)
                else False,
                "novelty_assessment": novelty.get("novelty_assessment", "")
                if isinstance(novelty, dict)
                else "",
                "query_plans": query_plans,
            },
        }
        ExecutiveReportAgent._assert_strict_schema(output)
        return output

    @staticmethod
    def _assert_strict_schema(report: Dict) -> None:
        expected_top_level_keys = {
            "paper_summary",
            "key_contributions",
            "key_assumptions",
            "identified_limitations",
            "improvement_opportunities",
            "generated_hypotheses",
            "required_resources",
            "estimated_experiment_cost_level",
            "potential_impact",
            "risk_factors",
            "hypothesis_critique",
            "proposed_experiments",
            "novelty_analysis",
        }
        actual_keys = set(report.keys())
        if actual_keys != expected_top_level_keys:
            missing = sorted(expected_top_level_keys - actual_keys)
            extra = sorted(actual_keys - expected_top_level_keys)
            raise RuntimeError(
                f"Strict schema validation failed. Missing keys: {missing}, Extra keys: {extra}"
            )

        if not isinstance(report["paper_summary"], str):
            raise RuntimeError("Strict schema validation failed: paper_summary must be string")
        for key in [
            "key_contributions",
            "key_assumptions",
            "identified_limitations",
            "improvement_opportunities",
            "generated_hypotheses",
            "required_resources",
            "risk_factors",
        ]:
            if not isinstance(report[key], list) or not all(
                isinstance(item, str) for item in report[key]
            ):
                raise RuntimeError(f"Strict schema validation failed: {key} must be list[str]")

        if report["estimated_experiment_cost_level"] not in {"Low", "Medium", "High"}:
            raise RuntimeError(
                "Strict schema validation failed: estimated_experiment_cost_level must be Low/Medium/High"
            )

        if not isinstance(report["potential_impact"], int) or not (1 <= report["potential_impact"] <= 5):
            raise RuntimeError(
                "Strict schema validation failed: potential_impact must be int in range 1-5"
            )

        if not isinstance(report["hypothesis_critique"], list):
            raise RuntimeError("Strict schema validation failed: hypothesis_critique must be list")
        for item in report["hypothesis_critique"]:
            required = {
                "hypothesis",
                "novelty",
                "testability",
                "mechanistic_clarity",
                "feasibility",
                "improvement_suggestions",
            }
            if not isinstance(item, dict) or set(item.keys()) != required:
                raise RuntimeError("Strict schema validation failed: invalid hypothesis_critique item")
            if not isinstance(item["hypothesis"], str):
                raise RuntimeError("Strict schema validation failed: hypothesis must be string")
            for score_key in ["novelty", "testability", "mechanistic_clarity", "feasibility"]:
                if not isinstance(item[score_key], float) or not (0.0 <= item[score_key] <= 1.0):
                    raise RuntimeError(
                        f"Strict schema validation failed: {score_key} must be float in range 0-1"
                    )
            if not isinstance(item["improvement_suggestions"], list) or not all(
                isinstance(s, str) for s in item["improvement_suggestions"]
            ):
                raise RuntimeError(
                    "Strict schema validation failed: improvement_suggestions must be list[str]"
                )

        if not isinstance(report["proposed_experiments"], list):
            raise RuntimeError("Strict schema validation failed: proposed_experiments must be list")
        for item in report["proposed_experiments"]:
            required = {
                "objective",
                "design",
                "required_resources",
                "evaluation_metrics",
                "risks",
                "estimated_experiment_cost_level",
                "potential_impact",
                "risk_factors",
            }
            if not isinstance(item, dict) or set(item.keys()) != required:
                raise RuntimeError("Strict schema validation failed: invalid proposed_experiments item")
            for key in ["objective", "design", "required_resources", "evaluation_metrics", "risks"]:
                if not isinstance(item[key], str):
                    raise RuntimeError(
                        f"Strict schema validation failed: proposed_experiments.{key} must be string"
                    )
            if item["estimated_experiment_cost_level"] not in {"Low", "Medium", "High"}:
                raise RuntimeError(
                    "Strict schema validation failed: proposed_experiments.estimated_experiment_cost_level must be Low/Medium/High"
                )
            if not isinstance(item["potential_impact"], int) or not (1 <= item["potential_impact"] <= 5):
                raise RuntimeError(
                    "Strict schema validation failed: proposed_experiments.potential_impact must be int in range 1-5"
                )
            if not isinstance(item["risk_factors"], list) or not all(
                isinstance(r, str) for r in item["risk_factors"]
            ):
                raise RuntimeError(
                    "Strict schema validation failed: proposed_experiments.risk_factors must be list[str]"
                )

        novelty = report["novelty_analysis"]
        required_novelty_keys = {
            "similar_papers_found",
            "recommended_papers",
            "novelty_score",
            "high_similarity_found",
            "novelty_assessment",
            "query_plans",
        }
        if not isinstance(novelty, dict) or set(novelty.keys()) != required_novelty_keys:
            raise RuntimeError("Strict schema validation failed: invalid novelty_analysis")
        if not isinstance(novelty["similar_papers_found"], bool):
            raise RuntimeError(
                "Strict schema validation failed: novelty_analysis.similar_papers_found must be bool"
            )
        if not isinstance(novelty["novelty_score"], float) or not (0.0 <= novelty["novelty_score"] <= 1.0):
            raise RuntimeError(
                "Strict schema validation failed: novelty_analysis.novelty_score must be float in range 0-1"
            )
        if not isinstance(novelty["high_similarity_found"], bool):
            raise RuntimeError(
                "Strict schema validation failed: novelty_analysis.high_similarity_found must be bool"
            )
        if not isinstance(novelty["novelty_assessment"], str):
            raise RuntimeError(
                "Strict schema validation failed: novelty_analysis.novelty_assessment must be string"
            )
        if not isinstance(novelty["recommended_papers"], list):
            raise RuntimeError(
                "Strict schema validation failed: novelty_analysis.recommended_papers must be list"
            )

    def run(self, context: Dict) -> Dict:
        chunks: List[str] = context.get("chunks", [])
        if not chunks:
            raise RuntimeError("No chunks available for analysis")

        paper_context = self._retrieve_context(chunks)
        state: Dict = {"paper_context": paper_context}

        state.update(self.summary_agent.run(state))
        state.update(self.limitation_agent.run(state))
        # Debate mode:
        # 1) generator proposes 3 hypotheses
        # 2) limitation agent critiques each
        # 3) generator revises
        initial_hypotheses_state = self.hypothesis_agent.run(state)
        state["initial_hypotheses"] = initial_hypotheses_state.get("generated_hypotheses", [])

        critique_state = self.limitation_agent.run(
            {**state, "hypotheses_to_critique": state.get("initial_hypotheses", [])}
        )
        state.update(critique_state)

        revised_hypotheses_state = self.hypothesis_agent.run(
            {
                **state,
                "initial_hypotheses": state.get("initial_hypotheses", []),
                "hypothesis_critiques": state.get("hypothesis_critiques", []),
            }
        )
        state.update(revised_hypotheses_state)
        state.update(self.hypothesis_critic_agent.run(state))
        state.update(self.experiment_agent.run(state))
        state.update(self.novelty_agent.run(state))

        return self._validate_schema(state)
