from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
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

    @staticmethod
    def _extract_with_macos_vision_ocr(pdf_path: str | Path) -> str:
        if platform.system() != "Darwin":
            return ""

        swift_code = r"""
import Foundation
import PDFKit
import Vision
import AppKit

let args = CommandLine.arguments
guard args.count >= 2 else {
    print("")
    exit(0)
}
let pdfPath = args[1]
guard let document = PDFDocument(url: URL(fileURLWithPath: pdfPath)) else {
    print("")
    exit(0)
}

func renderPageToCGImage(_ page: PDFPage, scale: CGFloat = 2.0) -> CGImage? {
    let rect = page.bounds(for: .mediaBox)
    let width = max(Int(rect.width * scale), 1)
    let height = max(Int(rect.height * scale), 1)
    guard let ctx = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }

    ctx.setFillColor(NSColor.white.cgColor)
    ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
    ctx.saveGState()
    ctx.translateBy(x: 0, y: CGFloat(height))
    ctx.scaleBy(x: scale, y: -scale)
    page.draw(with: .mediaBox, to: ctx)
    ctx.restoreGState()
    return ctx.makeImage()
}

var allText: [String] = []
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true

for i in 0..<document.pageCount {
    guard let page = document.page(at: i) else { continue }
    guard let cgImage = renderPageToCGImage(page) else { continue }
    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    do {
        try handler.perform([request])
        let observations = request.results as? [VNRecognizedTextObservation] ?? []
        let lines = observations.compactMap { $0.topCandidates(1).first?.string }
        if !lines.isEmpty {
            allText.append(lines.joined(separator: "\n"))
        }
    } catch {
        continue
    }
}

print(allText.joined(separator: "\n\n"))
"""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".swift", delete=False
            ) as tmp_swift:
                tmp_swift.write(swift_code)
                swift_path = tmp_swift.name

            result = subprocess.run(
                ["swift", swift_path, str(pdf_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            return (result.stdout or "").strip() if result.returncode == 0 else ""
        except Exception:
            return ""
        finally:
            try:
                Path(swift_path).unlink(missing_ok=True)  # type: ignore[name-defined]
            except Exception:
                pass

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

        # Fallback parser: macOS Spotlight text extraction.
        if not pages:
            try:
                result = subprocess.run(
                    ["mdls", "-name", "kMDItemTextContent", "-raw", str(pdf_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    mdls_text = (result.stdout or "").strip()
                    if mdls_text and mdls_text != "(null)":
                        pages.append(mdls_text)
                    else:
                        parser_errors.append("mdls returned no text content")
                else:
                    stderr = (result.stderr or "").strip()
                    parser_errors.append(f"mdls failed: {stderr or 'unknown error'}")
            except Exception as exc:
                parser_errors.append(f"mdls unavailable: {exc}")

        # Fallback parser: textutil conversion on macOS.
        if not pages:
            try:
                result = subprocess.run(
                    ["textutil", "-convert", "txt", "-stdout", str(pdf_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    textutil_text = (result.stdout or "").strip()
                    if textutil_text:
                        pages.append(textutil_text)
                    else:
                        parser_errors.append("textutil returned empty output")
                else:
                    stderr = (result.stderr or "").strip()
                    parser_errors.append(f"textutil failed: {stderr or 'unknown error'}")
            except Exception as exc:
                parser_errors.append(f"textutil unavailable: {exc}")

        # Fallback parser: macOS Vision OCR for scanned PDFs.
        if not pages:
            ocr_text = self._extract_with_macos_vision_ocr(pdf_path)
            if ocr_text:
                pages.append(ocr_text)
            else:
                parser_errors.append("Vision OCR returned no text content")

        text = "\n\n".join(pages).strip()
        if not text:
            details = "; ".join(parser_errors)
            raise RuntimeError(
                "No extractable text found in PDF. Install parser dependency "
                "(`pip install pymupdf` or `pip install pdfplumber`) "
                "or use macOS built-in extractors (mdls/textutil). "
                f"Python executable: {sys.executable}. "
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
