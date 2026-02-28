from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import tiktoken

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False


@dataclass
class RetrievedChunk:
    chunk_id: int
    text: str
    score: float


class FAISSVectorStore:
    def __init__(self) -> None:
        self.index: object | None = None
        self.embedding_matrix: np.ndarray | None = None
        self.chunk_map: Dict[int, str] = {}
        self.tokenized_chunks: Dict[int, List[str]] = {}
        self.doc_lengths: Dict[int, int] = {}
        self.term_doc_freq: Dict[str, int] = defaultdict(int)
        self.term_freq_by_doc: Dict[int, Counter[str]] = {}
        self.avg_doc_len: float = 0.0
        self.total_docs: int = 0
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoding = None
        self.last_retrieval_diagnostics: Dict = {}
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _to_normalized_np(vectors: List[List[float]]) -> np.ndarray:
        arr = np.array(vectors, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9]+", text.lower())

    def _build_sparse_index(self, chunks: List[str]) -> None:
        self.term_doc_freq = defaultdict(int)
        self.term_freq_by_doc = {}
        self.doc_lengths = {}
        self.tokenized_chunks = {}

        total_len = 0
        for idx, text in enumerate(chunks):
            tokens = self._tokenize(text)
            self.tokenized_chunks[idx] = tokens
            doc_tf = Counter(tokens)
            self.term_freq_by_doc[idx] = doc_tf
            doc_len = len(tokens)
            self.doc_lengths[idx] = doc_len
            total_len += doc_len

            for term in doc_tf:
                self.term_doc_freq[term] += 1

        self.total_docs = len(chunks)
        self.avg_doc_len = total_len / max(self.total_docs, 1)

    def build(self, embeddings: List[List[float]], chunks: List[str]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must have the same length")
        if not embeddings:
            raise ValueError("Cannot build vector store with empty embeddings")

        emb = self._to_normalized_np(embeddings)
        self.embedding_matrix = emb
        if FAISS_AVAILABLE:
            dim = emb.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(emb)
        else:
            self.index = None
            self.logger.warning(
                "faiss is not installed. Falling back to NumPy dense retrieval."
            )
        self.chunk_map = {idx: text for idx, text in enumerate(chunks)}
        self._build_sparse_index(chunks)

    def _dense_retrieve(
        self,
        query_embedding: List[float],
        top_n: int,
    ) -> List[Tuple[int, float]]:
        if self.embedding_matrix is None:
            raise RuntimeError("Vector store not initialized")

        query = np.array([query_embedding], dtype="float32")
        qnorm = np.linalg.norm(query, axis=1, keepdims=True)
        qnorm[qnorm == 0] = 1.0
        query = query / qnorm

        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(query, top_n)
            dense_ranked: List[Tuple[int, float]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                dense_ranked.append((int(idx), float(score)))
            return dense_ranked

        dense_scores = self.embedding_matrix @ query[0]
        if dense_scores.size == 0:
            return []
        top_n = min(top_n, dense_scores.shape[0])
        top_indices = np.argsort(-dense_scores)[:top_n]
        return [(int(idx), float(dense_scores[idx])) for idx in top_indices]

    def _idf(self, term: str) -> float:
        df = self.term_doc_freq.get(term, 0)
        # BM25-style smoothed IDF.
        return math.log(1 + (self.total_docs - df + 0.5) / (df + 0.5))

    def _sparse_retrieve(self, query_text: str, top_n: int) -> List[Tuple[int, float]]:
        if not query_text.strip() or self.total_docs == 0:
            return []

        query_terms = self._tokenize(query_text)
        if not query_terms:
            return []

        k1 = 1.5
        b = 0.75
        scores: Dict[int, float] = defaultdict(float)

        for term in query_terms:
            idf = self._idf(term)
            if idf <= 0:
                continue
            for doc_id, tf_map in self.term_freq_by_doc.items():
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths.get(doc_id, 0)
                norm = k1 * (1 - b + b * (doc_len / max(self.avg_doc_len, 1.0)))
                bm25_term = idf * ((tf * (k1 + 1)) / (tf + norm))
                scores[doc_id] += bm25_term

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    @staticmethod
    def _rrf_fuse(
        dense_ranked: List[Tuple[int, float]],
        sparse_ranked: List[Tuple[int, float]],
        rrf_k: int = 60,
    ) -> List[Tuple[int, float]]:
        fused: Dict[int, float] = defaultdict(float)
        for rank, (doc_id, _) in enumerate(dense_ranked, start=1):
            fused[doc_id] += 1.0 / (rrf_k + rank)
        for rank, (doc_id, _) in enumerate(sparse_ranked, start=1):
            fused[doc_id] += 1.0 / (rrf_k + rank)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    def retrieve(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        query_text: str = "",
        max_context_tokens: int = 16000,
    ) -> List[RetrievedChunk]:
        if self.embedding_matrix is None:
            raise RuntimeError("Vector store not initialized")

        candidate_pool = max(top_k * 4, 20)
        dense_ranked = self._dense_retrieve(query_embedding, candidate_pool)
        sparse_ranked = self._sparse_retrieve(query_text, candidate_pool)
        fused_ranked = self._rrf_fuse(dense_ranked, sparse_ranked, rrf_k=60)

        results: List[RetrievedChunk] = []
        used_tokens = 0
        dedup_ids = set()
        for idx, score in fused_ranked:
            if idx in dedup_ids:
                continue
            dedup_ids.add(idx)
            text = self.chunk_map.get(int(idx), "")
            if not text:
                continue
            if self.encoding is not None:
                chunk_tokens = len(self.encoding.encode(text))
            else:
                chunk_tokens = len(text.split())
            if used_tokens + chunk_tokens > max_context_tokens:
                continue
            used_tokens += chunk_tokens
            results.append(
                RetrievedChunk(chunk_id=int(idx), text=text, score=float(score))
            )
            if len(results) >= top_k:
                break

        self.last_retrieval_diagnostics = {
            "dense_candidates": len(dense_ranked),
            "sparse_candidates": len(sparse_ranked),
            "fused_candidates": len(fused_ranked),
            "deduplicated_results": len(results),
            "context_tokens_used": used_tokens,
            "context_token_limit": max_context_tokens,
            "query_text_used": bool(query_text.strip()),
            "tokenizer_fallback_used": self.encoding is None,
            "faiss_available": FAISS_AVAILABLE,
        }
        self.logger.info("Hybrid retrieval diagnostics: %s", self.last_retrieval_diagnostics)

        return results
