from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

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
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.encoding = None

    @staticmethod
    def _simple_encode(text: str) -> List[str]:
        return text.split()

    @staticmethod
    def _simple_decode(tokens: List[str]) -> str:
        return " ".join(tokens)

    def extract_text(self, pdf_path: str | Path) -> str:
        pages: List[str] = []
        parser_errors: List[str] = []

        # Primary parser: PyMuPDF (fitz)
        try:
            import fitz  # type: ignore

            try:
                doc = fitz.open(str(pdf_path))
                try:
                    for page in doc:
                        page_text = page.get_text("text") or ""
                        if page_text.strip():
                            pages.append(page_text)
                finally:
                    doc.close()
            except Exception as exc:
                parser_errors.append(f"PyMuPDF failed: {exc}")
        except Exception:
            parser_errors.append("PyMuPDF (fitz) is not installed")

        # Fallback parser: pdfplumber
        if not pages:
            try:
                import pdfplumber  # type: ignore

                try:
                    with pdfplumber.open(str(pdf_path)) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text() or ""
                            if page_text.strip():
                                pages.append(page_text)
                except Exception as exc:
                    parser_errors.append(f"pdfplumber failed: {exc}")
            except Exception:
                parser_errors.append("pdfplumber is not installed")

        text = "\n\n".join(pages).strip()
        if not text:
            details = "; ".join(parser_errors)
            raise RuntimeError(
                "No extractable text found in PDF. Install parser dependency "
                "(`pip install pymupdf` or `pip install pdfplumber`). "
                f"Debug details: {details}"
            )
        return text

    def chunk_text(self, text: str) -> List[PDFChunk]:
        chunks: List[PDFChunk] = []
        step = self.chunk_size_tokens - self.chunk_overlap_tokens

        if self.encoding is not None:
            tokens = self.encoding.encode(text)
            if not tokens:
                return []

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

        # Offline-safe fallback when tiktoken cannot initialize.
        fallback_tokens = self._simple_encode(text)
        if not fallback_tokens:
            return []
        start = 0
        chunk_id = 0
        while start < len(fallback_tokens):
            end = min(start + self.chunk_size_tokens, len(fallback_tokens))
            chunk_text = self._simple_decode(fallback_tokens[start:end]).strip()
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
