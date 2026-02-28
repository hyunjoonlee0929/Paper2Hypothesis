from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from agents import (
    ExecutiveReportAgent,
    ExperimentDesignAgent,
    HypothesisCriticAgent,
    HypothesisGeneratorAgent,
    LimitationAgent,
    NoveltyCheckAgent,
    SummaryAgent,
)
from core.embedding import OpenAIEmbedder
from core.literature_search import SemanticScholarClient
from core.llm_client import LLMClient
from core.pdf_processor import PDFProcessor
from core.vector_store import FAISSVectorStore

load_dotenv()

st.set_page_config(page_title="Paper2Hypothesis", layout="wide")
st.title("Paper2Hypothesis MVP")
st.caption("Upload one paper PDF and generate structured hypothesis-driven analysis.")

with st.sidebar:
    st.header("Configuration")
    user_api_key = st.text_input("OpenAI API Key", type="password")
    model_name = st.text_input("LLM Model", value="gpt-4o-mini")
    embedding_model = st.text_input("Embedding Model", value="text-embedding-3-small")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    mock_mode = st.checkbox("Mock mode (no LLM/API calls)", value=False)
    st.divider()
    st.caption(f"Python: `{sys.executable}`")
    fitz_ok = importlib.util.find_spec("fitz") is not None
    pdfplumber_ok = importlib.util.find_spec("pdfplumber") is not None
    st.caption(f"fitz installed: `{fitz_ok}`")
    st.caption(f"pdfplumber installed: `{pdfplumber_ok}`")

uploaded_file = st.file_uploader("Upload a paper PDF", type=["pdf"])


def render_section(title: str, content) -> None:
    st.subheader(title)
    if isinstance(content, list):
        if content and isinstance(content[0], dict):
            st.json(content)
            return
        for item in content:
            st.markdown(f"- {item}")
    elif isinstance(content, dict):
        st.json(content)
    else:
        st.write(content)


def format_runtime_error(message: str) -> str:
    if message.startswith("OPENAI_INSUFFICIENT_QUOTA:"):
        return (
            "OpenAI quota is exhausted for this API key.\n"
            "1) Check billing/usage at https://platform.openai.com/usage\n"
            "2) Add payment method or increase budget at https://platform.openai.com/settings/organization/billing\n"
            "3) Retry with a funded API key."
        )
    if message.startswith("OPENAI_INVALID_API_KEY:"):
        return (
            "Invalid OpenAI API key.\n"
            "Please provide a valid key in the sidebar or set OPENAI_API_KEY correctly."
        )
    if message.startswith("OPENAI_RATE_LIMIT:"):
        return "OpenAI rate limit hit. Wait briefly and retry, or lower request volume."
    return message


def build_mock_report() -> dict:
    return {
        "paper_summary": "This is a mock summary for UI testing.",
        "key_contributions": [
            "Introduces a new methodological framework.",
            "Reports improved benchmark performance.",
        ],
        "key_assumptions": [
            "Training and test distributions are reasonably aligned.",
            "Observed gains are not solely due to larger model capacity.",
        ],
        "identified_limitations": [
            "Limited external validity across domains.",
            "Ablation coverage is incomplete for key components.",
        ],
        "improvement_opportunities": [
            "Add stronger out-of-distribution evaluation.",
            "Run controlled studies for each architectural choice.",
        ],
        "generated_hypotheses": [
            "Explicit causal regularization will improve robustness under shift.",
            "Adaptive curriculum on hard negatives will improve sample efficiency.",
            "Mechanism-aware augmentation will reduce failure modes in edge cases.",
        ],
        "required_resources": [
            "2-4 GPUs",
            "Benchmark datasets with OOD splits",
            "Experiment tracking infrastructure",
        ],
        "estimated_experiment_cost_level": "Medium",
        "potential_impact": 4,
        "risk_factors": [
            "Confounding from data leakage",
            "Compute budget overrun",
            "Inconclusive causality attribution",
        ],
        "hypothesis_critique": [
            {
                "hypothesis": "Explicit causal regularization will improve robustness under shift.",
                "novelty": 0.67,
                "testability": 0.86,
                "mechanistic_clarity": 0.74,
                "feasibility": 0.8,
                "improvement_suggestions": ["Specify causal graph assumptions."],
            },
            {
                "hypothesis": "Adaptive curriculum on hard negatives will improve sample efficiency.",
                "novelty": 0.58,
                "testability": 0.91,
                "mechanistic_clarity": 0.7,
                "feasibility": 0.88,
                "improvement_suggestions": ["Define curriculum schedule and stopping criteria."],
            },
            {
                "hypothesis": "Mechanism-aware augmentation will reduce failure modes in edge cases.",
                "novelty": 0.71,
                "testability": 0.79,
                "mechanistic_clarity": 0.77,
                "feasibility": 0.73,
                "improvement_suggestions": ["Pre-register failure taxonomy before experiments."],
            },
        ],
        "proposed_experiments": [
            {
                "objective": "Measure robustness under controlled distribution shift.",
                "design": "Compare baseline vs intervention across 3 OOD benchmarks.",
                "required_resources": "4 GPUs, 2 weeks runtime, benchmark datasets",
                "evaluation_metrics": "OOD accuracy, calibration error, robustness gap",
                "risks": "Potential confounding from preprocessing differences",
                "estimated_experiment_cost_level": "Medium",
                "potential_impact": 4,
                "risk_factors": ["Reproducibility variance", "Data mismatch"],
            }
        ],
        "novelty_analysis": {
            "similar_papers_found": True,
            "recommended_papers": [
                {
                    "title": "Mock Paper A",
                    "authors": "A. Researcher, B. Scientist",
                    "year": "2024",
                    "abstract": "Mock abstract for UI testing.",
                    "semantic_similarity": 0.62,
                    "is_high_similarity": False,
                    "matched_hypothesis": "Explicit causal regularization will improve robustness under shift.",
                }
            ],
            "novelty_score": 0.38,
            "high_similarity_found": False,
            "novelty_assessment": "Moderately novel relative to closely related prior work.",
            "query_plans": [
                {
                    "hypothesis": "Explicit causal regularization will improve robustness under shift.",
                    "keyword_triple": {
                        "subject": "causal regularization",
                        "relation": "improves",
                        "mechanism": "distribution shift robustness",
                    },
                    "exact_phrase_query": "\"causal regularization distribution shift robustness\"",
                    "keyword_and_query": "causal regularization AND robustness AND OOD",
                    "mechanism_focused_query": "mechanism causal regularization robustness shift",
                }
            ],
        },
    }


if st.button("Run Analysis", type="primary"):
    if uploaded_file is None:
        st.error("Please upload a PDF first.")
        st.stop()

    api_key = user_api_key.strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not mock_mode and not api_key:
        st.error("Please provide OPENAI_API_KEY via sidebar or environment variable.")
        st.stop()

    try:
        if mock_mode:
            with st.spinner("Running in mock mode (no API calls)..."):
                report = build_mock_report()
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_pdf_path = Path(tmp.name)

            pdf_processor = PDFProcessor(chunk_size_tokens=1200, chunk_overlap_tokens=150)
            chunks = [chunk.text for chunk in pdf_processor.process(tmp_pdf_path)]

            openai_client = OpenAI(api_key=api_key)
            llm_client = LLMClient(client=openai_client, model=model_name, temperature=temperature)
            embedder = OpenAIEmbedder(client=openai_client, model=embedding_model)
            vector_store = FAISSVectorStore()
            s2_client = SemanticScholarClient()

            summary_agent = SummaryAgent(llm_client)
            limitation_agent = LimitationAgent(llm_client)
            hypothesis_agent = HypothesisGeneratorAgent(llm_client)
            hypothesis_critic_agent = HypothesisCriticAgent(llm_client)
            experiment_agent = ExperimentDesignAgent(llm_client)
            novelty_agent = NoveltyCheckAgent(llm_client, s2_client, embedder)

            orchestrator = ExecutiveReportAgent(
                summary_agent=summary_agent,
                limitation_agent=limitation_agent,
                hypothesis_agent=hypothesis_agent,
                hypothesis_critic_agent=hypothesis_critic_agent,
                experiment_agent=experiment_agent,
                novelty_agent=novelty_agent,
                embedder=embedder,
                vector_store=vector_store,
            )

            with st.spinner("Running multi-agent analysis..."):
                report = orchestrator.run({"chunks": chunks})

        st.success("Analysis complete.")

        st.subheader("Structured JSON Output")
        st.code(json.dumps(report, indent=2, ensure_ascii=False), language="json")

        render_section("Paper Summary", report.get("paper_summary", ""))
        render_section("Key Contributions", report.get("key_contributions", []))
        render_section("Key Assumptions", report.get("key_assumptions", []))
        render_section("Identified Limitations", report.get("identified_limitations", []))
        render_section("Improvement Opportunities", report.get("improvement_opportunities", []))
        render_section("Generated Hypotheses", report.get("generated_hypotheses", []))
        render_section("Required Resources", report.get("required_resources", []))
        render_section(
            "Estimated Experiment Cost Level", report.get("estimated_experiment_cost_level", "")
        )
        render_section("Potential Impact (1-5)", report.get("potential_impact", 1))
        render_section("Risk Factors", report.get("risk_factors", []))
        render_section("Hypothesis Critique", report.get("hypothesis_critique", []))
        render_section("Proposed Experiments", report.get("proposed_experiments", []))
        render_section("Novelty Analysis", report.get("novelty_analysis", {}))

    except RuntimeError as exc:
        st.error(format_runtime_error(str(exc)))
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
    finally:
        if "tmp_pdf_path" in locals() and tmp_pdf_path.exists():
            tmp_pdf_path.unlink(missing_ok=True)

# Future expansion ideas (MVP out of scope):
# - Multi-paper cross-analysis
# - Citation graph analysis
# - Self-critique loop
# - Hypothesis scoring model
