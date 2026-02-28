from __future__ import annotations

import json
import os
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


if st.button("Run Analysis", type="primary"):
    if uploaded_file is None:
        st.error("Please upload a PDF first.")
        st.stop()

    api_key = user_api_key.strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        st.error("Please provide OPENAI_API_KEY via sidebar or environment variable.")
        st.stop()

    try:
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
        st.error(str(exc))
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
