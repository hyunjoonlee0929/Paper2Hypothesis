from __future__ import annotations

from typing import Any, Dict, List

import requests


class SemanticScholarClient:
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, timeout: int = 20) -> None:
        self.timeout = timeout

    def search_papers(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,year,abstract,authors",
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Semantic Scholar API error: {exc}") from exc

        papers: List[Dict[str, Any]] = []
        for item in payload.get("data", []):
            authors = ", ".join(a.get("name", "") for a in item.get("authors", []) if a.get("name"))
            papers.append(
                {
                    "title": item.get("title", ""),
                    "authors": authors,
                    "year": str(item.get("year", "")),
                    "abstract": item.get("abstract", "") or "",
                }
            )

        return papers
