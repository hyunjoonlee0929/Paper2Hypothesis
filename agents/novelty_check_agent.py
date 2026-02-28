from __future__ import annotations

from typing import Dict, List

import numpy as np

from core.embedding import OpenAIEmbedder
from core.literature_search import SemanticScholarClient
from core.llm_client import LLMClient


class NoveltyCheckAgent:
    def __init__(
        self,
        llm: LLMClient,
        s2_client: SemanticScholarClient,
        embedder: OpenAIEmbedder,
    ) -> None:
        self.llm = llm
        self.s2_client = s2_client
        self.embedder = embedder

    @staticmethod
    def _normalize(vectors: List[List[float]]) -> np.ndarray:
        arr = np.array(vectors, dtype="float32")
        if arr.size == 0:
            return arr
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def _build_query_plan(self, hypotheses: List[str]) -> List[Dict[str, str]]:
        if not hypotheses:
            return []

        prompt = f"""
Create a structured keyword triple and search-query plan for each hypothesis.
Return JSON with key:
- hypothesis_queries: array of objects

Each object must contain:
- hypothesis
- keyword_triple: object with keys subject, relation, mechanism
- exact_phrase_query
- keyword_and_query
- mechanism_focused_query

Hypotheses:
{hypotheses}
""".strip()

        result = self.llm.generate_json(prompt)
        raw_items = result.get("hypothesis_queries", [])
        plans: List[Dict[str, str]] = []

        for item in raw_items if isinstance(raw_items, list) else []:
            if not isinstance(item, dict):
                continue
            triple = item.get("keyword_triple", {})
            if not isinstance(triple, dict):
                triple = {}
            hypothesis = item.get("hypothesis", "")
            plans.append(
                {
                    "hypothesis": hypothesis,
                    "subject": str(triple.get("subject", "")).strip(),
                    "relation": str(triple.get("relation", "")).strip(),
                    "mechanism": str(triple.get("mechanism", "")).strip(),
                    "exact_phrase_query": str(item.get("exact_phrase_query", f'"{hypothesis}"')).strip(),
                    "keyword_and_query": str(item.get("keyword_and_query", "")).strip(),
                    "mechanism_focused_query": str(item.get("mechanism_focused_query", "")).strip(),
                }
            )

        if plans:
            return plans

        # Fallback if LLM output is malformed.
        for hypothesis in hypotheses:
            parts = hypothesis.split()
            subject = " ".join(parts[:4]).strip()
            mechanism = " ".join(parts[4:8]).strip()
            plans.append(
                {
                    "hypothesis": hypothesis,
                    "subject": subject,
                    "relation": "influences",
                    "mechanism": mechanism,
                    "exact_phrase_query": f'"{hypothesis}"',
                    "keyword_and_query": f"{subject} AND {mechanism}".strip(),
                    "mechanism_focused_query": f"mechanism {mechanism} {subject}".strip(),
                }
            )
        return plans

    def _search_multi_query(self, plans: List[Dict[str, str]]) -> List[Dict]:
        deduped: Dict[str, Dict] = {}
        first_error: RuntimeError | None = None

        for plan in plans:
            queries = [
                plan.get("exact_phrase_query", ""),
                plan.get("keyword_and_query", ""),
                plan.get("mechanism_focused_query", ""),
            ]
            for query in queries:
                if not query:
                    continue
                try:
                    papers = self.s2_client.search_papers(query=query, limit=5)
                except RuntimeError as exc:
                    if first_error is None:
                        first_error = exc
                    continue

                for paper in papers:
                    key = f"{paper.get('title', '').strip().lower()}::{paper.get('year', '')}"
                    if not key.strip(":"):
                        continue
                    if key not in deduped:
                        deduped[key] = paper

        if not deduped and first_error is not None:
            raise first_error

        return list(deduped.values())

    def _attach_semantic_similarity(
        self, hypotheses: List[str], papers: List[Dict], high_similarity_threshold: float = 0.75
    ) -> Dict:
        if not hypotheses or not papers:
            return {
                "papers": papers,
                "max_similarity": 0.0,
                "high_similarity_found": False,
            }

        hypothesis_embeddings = self._normalize(self.embedder.embed_texts(hypotheses))
        abstract_texts = [paper.get("abstract", "") or paper.get("title", "") for paper in papers]
        abstract_embeddings = self._normalize(self.embedder.embed_texts(abstract_texts))

        if hypothesis_embeddings.size == 0 or abstract_embeddings.size == 0:
            return {
                "papers": papers,
                "max_similarity": 0.0,
                "high_similarity_found": False,
            }

        similarity_matrix = abstract_embeddings @ hypothesis_embeddings.T
        max_by_paper = similarity_matrix.max(axis=1)
        best_hypothesis_idx = similarity_matrix.argmax(axis=1)

        enriched = []
        high_similarity_found = False
        for idx, paper in enumerate(papers):
            sim = float(max_by_paper[idx])
            is_high = sim > high_similarity_threshold
            high_similarity_found = high_similarity_found or is_high
            enriched.append(
                {
                    **paper,
                    "semantic_similarity": round(sim, 4),
                    "is_high_similarity": is_high,
                    "matched_hypothesis": hypotheses[int(best_hypothesis_idx[idx])],
                }
            )

        max_similarity = float(max(max_by_paper)) if len(max_by_paper) > 0 else 0.0
        return {
            "papers": enriched,
            "max_similarity": max_similarity,
            "high_similarity_found": high_similarity_found,
        }

    def run(self, context: Dict) -> Dict:
        hypotheses: List[str] = context.get("generated_hypotheses", [])
        if not hypotheses:
            return {
                "novelty_analysis": {
                    "similar_papers_found": False,
                    "recommended_papers": [],
                    "novelty_score": 0.0,
                    "high_similarity_found": False,
                    "novelty_assessment": "No hypotheses provided for novelty check.",
                }
            }

        query_plans = self._build_query_plan(hypotheses)
        papers = self._search_multi_query(query_plans)

        sim_info = self._attach_semantic_similarity(hypotheses, papers, high_similarity_threshold=0.75)
        recommended_papers = sim_info["papers"]
        max_similarity = sim_info["max_similarity"]
        novelty_score = max(0.0, min(1.0, round(1.0 - max_similarity, 4)))

        prompt = f"""
Assess novelty of these hypotheses using retrieved papers and semantic similarities.
Return JSON with keys:
- similar_papers_found (boolean)
- novelty_assessment (string)

Hypotheses:
{hypotheses}

Retrieved papers with similarity:
{recommended_papers}

Novelty score (computed): {novelty_score}
High similarity threshold: 0.75
""".strip()

        llm_result = self.llm.generate_json(prompt)
        novelty_analysis = {
            "similar_papers_found": bool(llm_result.get("similar_papers_found", bool(recommended_papers))),
            "recommended_papers": recommended_papers,
            "novelty_score": novelty_score,
            "high_similarity_found": bool(sim_info["high_similarity_found"]),
            "novelty_assessment": llm_result.get("novelty_assessment", ""),
            "query_plans": [
                {
                    "hypothesis": plan.get("hypothesis", ""),
                    "keyword_triple": {
                        "subject": plan.get("subject", ""),
                        "relation": plan.get("relation", ""),
                        "mechanism": plan.get("mechanism", ""),
                    },
                    "exact_phrase_query": plan.get("exact_phrase_query", ""),
                    "keyword_and_query": plan.get("keyword_and_query", ""),
                    "mechanism_focused_query": plan.get("mechanism_focused_query", ""),
                }
                for plan in query_plans
            ],
        }
        return {"novelty_analysis": novelty_analysis}
