from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import tiktoken


@dataclass
class PDFChunk:
    chunk_id: int
    text: str


class PDFProcessor:
    def __init__(
        self,
        chunk_size_tokens: int = 1200,
        chunk_overlap_tokens: int = 150,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens cannot be negative")
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)

    def extract_text(self, pdf_path: str | Path) -> str:
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            raise RuntimeError(f"Failed to open PDF: {exc}") from exc

        pages: List[str] = []
        try:
            for page in doc:
                page_text = page.get_text("text") or ""
                if page_text.strip():
                    pages.append(page_text)
        finally:
            doc.close()

        text = "\n\n".join(pages).strip()
        if not text:
            raise RuntimeError("No extractable text found in PDF")
        return text

    def chunk_text(self, text: str) -> List[PDFChunk]:
        tokens = self.encoding.encode(text)
        if not tokens:
            return []

        chunks: List[PDFChunk] = []
        step = self.chunk_size_tokens - self.chunk_overlap_tokens

        start = 0
        chunk_id = 0
        while start < len(tokens):
            end = min(start + self.chunk_size_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens).strip()
            if chunk_text:
                chunks.append(PDFChunk(chunk_id=chunk_id, text=chunk_text))
                chunk_id += 1
            start += step

        return chunks

    def process(self, pdf_path: str | Path) -> List[PDFChunk]:
        text = self.extract_text(pdf_path)
        chunks = self.chunk_text(text)
        if not chunks:
            raise RuntimeError("Failed to create chunks from extracted text")
        return chunks
