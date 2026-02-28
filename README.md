# Paper2Hypothesis MVP

Paper2Hypothesis is a local Streamlit app that analyzes a single research paper PDF and generates a structured JSON report with:
- summary and key contributions
- methodological limitations and improvements
- new hypotheses
- experiment designs
- novelty check via Semantic Scholar
- strict schema-validated structured JSON output

## Project structure

```text
Paper2Hypothesis/
  app.py
  requirements.txt
  core/
    pdf_processor.py
    embedding.py
    vector_store.py
    llm_client.py
    literature_search.py
  agents/
    summary_agent.py
    limitation_agent.py
    hypothesis_generator_agent.py
    hypothesis_critic_agent.py
    experiment_design_agent.py
    novelty_check_agent.py
    executive_report_agent.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Set API key:

```bash
export OPENAI_API_KEY="your_api_key"
```

## Run

```bash
source .venv/bin/activate
.venv/bin/python -m streamlit run app.py
```

Then:
1. Upload one PDF.
2. Choose mode in sidebar:
   - Normal mode: provide OpenAI API key
   - Mock mode: no API calls, returns sample schema-valid JSON
3. Click `Run Analysis`.

## Mock mode (UI testing without quota)

- Enable `Mock mode (no LLM/API calls)` in sidebar.
- Works without OpenAI quota.
- Skips LLM, embedding, and literature API calls.
- Returns deterministic sample JSON matching the strict output schema.

## Key implementation updates

- Debate mode:
  - `HypothesisGeneratorAgent` proposes 3 hypotheses
  - `LimitationAgent` critiques each
  - `HypothesisGeneratorAgent` revises
  - `HypothesisCriticAgent` scores final set
- Novelty check upgrade:
  - keyword triple extraction
  - multi-query Semantic Scholar search
  - semantic similarity scoring
  - `novelty_score` and `high_similarity_found`
- Retrieval upgrade:
  - hybrid dense + BM25 retrieval
  - Reciprocal Rank Fusion
  - deduplication
  - retrieval diagnostics
  - context budget cap (16k tokens)
- Robust runtime behavior:
  - `faiss` optional (NumPy fallback retrieval)
  - PDF extraction fallbacks (`pymupdf`, `pdfplumber`, `mdls`, `textutil`, Vision OCR attempt)
  - clearer OpenAI error messages for quota/key/rate-limit

## Output schema

The app enforces this JSON structure:

```json
{
  "paper_summary": "...",
  "key_contributions": [],
  "key_assumptions": [],
  "identified_limitations": [],
  "improvement_opportunities": [],
  "generated_hypotheses": [],
  "required_resources": [],
  "estimated_experiment_cost_level": "Medium",
  "potential_impact": 3,
  "risk_factors": [],
  "hypothesis_critique": [
    {
      "hypothesis": "",
      "novelty": 0.0,
      "testability": 0.0,
      "mechanistic_clarity": 0.0,
      "feasibility": 0.0,
      "improvement_suggestions": []
    }
  ],
  "proposed_experiments": [
    {
      "objective": "",
      "design": "",
      "required_resources": "",
      "evaluation_metrics": "",
      "risks": "",
      "estimated_experiment_cost_level": "Medium",
      "potential_impact": 3,
      "risk_factors": []
    }
  ],
  "novelty_analysis": {
    "similar_papers_found": true,
    "novelty_score": 0.0,
    "high_similarity_found": false,
    "recommended_papers": [
      {
        "title": "",
        "authors": "",
        "year": "",
        "abstract": "",
        "semantic_similarity": 0.0,
        "is_high_similarity": false,
        "matched_hypothesis": ""
      }
    ],
    "query_plans": [],
    "novelty_assessment": ""
  }
}
```

Top-level schema is strictly validated in `agents/executive_report_agent.py`.

## Notes

- Single-paper processing only (MVP scope).
- No async, no DB, no deployment config.
- LLM JSON parsing failures retry once in `core/llm_client.py`.
- On Python 3.12, `faiss-cpu` is skipped by marker and NumPy fallback is used.
