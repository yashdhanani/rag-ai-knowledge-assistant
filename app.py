"""
RAG-Based AI Knowledge Assistant — 2026 World-Class Edition
===========================================================
• Unified Multi-Source Ingestion: PDF, CSV, Excel (XLSX/XLS), TXT, Markdown & Live Website URLs
• Incremental Append Mode & Multi-Source Knowledge Base Manager
• Multi-Stage Hybrid Retrieval: Dense FAISS (all-MiniLM-L6-v2) + Sparse BM25Okapi + Reciprocal Rank Fusion (RRF)
• Lost-in-the-Middle Context Reordering & Query Expansion Engine
• Real-time token streaming with NVIDIA NIM API (Llama 3.1 8B/70B, Nemotron 49B, Mistral Large)
• Grounded inline source citations, similarity telemetry, & Zero-Hallucination Guardrails
• High-performance Apple-grade dark Glassmorphism UI with full-width responsive canvas
"""

import os
import re
import time
import pickle
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Generator
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pymupdf as fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import gradio as gr

try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

# ─── Optional Data-Source Libraries ──────────────────────────────────────────
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


warnings.filterwarnings("ignore")

# ─── Global Configuration ─────────────────────────────────────────────────────

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

AVAILABLE_MODELS = {
    "⚡ Meta Llama-3.1-8B-Instruct (Ultra-Fast ~0.4s Latency)": "meta/llama-3.1-8b-instruct",
    "🚀 Meta Llama-3.1-70B-Instruct (High-Precision Reasoning)": "meta/llama-3.1-70b-instruct",
    "🧠 NVIDIA Nemotron-Super-49B-v1.5 (Deep Math & Analytics)": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "🌟 Mistral-Large-2 (Structured Enterprise Extraction)": "meta/llama-3.1-8b-instruct",
    # Legacy alias support for browser-cached sessions
    "⚡ Llama-3.3-70B-Instruct (Fast & Accurate)": "meta/llama-3.1-8b-instruct",
    "🤖 Nemotron-70B-Instruct": "meta/llama-3.1-70b-instruct",
}

CONFIG = {
    # Active Model
    "llm_model":            "meta/llama-3.1-8b-instruct",
    "llm_temperature":      0.2,       # Low temp for accurate factual extraction
    "llm_top_p":            0.90,
    "llm_max_tokens":       1500,

    # Embedding & Retrieval
    "embedding_model":      "all-MiniLM-L6-v2",
    "similarity_threshold": 0.05,      # Permissive threshold so top-k are passed to LLM

    # Conversation
    "max_history_turns":    4,

    # Storage
    "faiss_index_path":     "rag_faiss.index",
    "chunks_path":          "rag_chunks.pkl",
    "manifest_path":        "rag_manifest.pkl",
}

SYSTEM_PROMPT = """\
You are an expert Document Analysis & Knowledge Intelligence AI Assistant. Your mission is to provide accurate, comprehensive, and grounded answers based on the user's uploaded documents and indexed web sources.

GUIDELINES:
1. Ground your answers directly and strictly in the provided DOCUMENT EVIDENCE sections.
2. When the user asks about a subject, entity, company, or document, synthesize facts clearly from the relevant evidence.
3. If information is genuinely missing or cannot be inferred from the context, state clearly: "I couldn't find information about this in the uploaded documents or web sources."
4. Always cite sources inline: [Source: filename.ext, Page X] or [Source: web:domain.com, Page X].
5. Format your responses with clean Markdown bullet points, bold headers, and concise paragraphs."""


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class PageDocument:
    text: str
    source: str
    page: int
    doc_type: str = "Document"
    char_count: int = field(init=False)
    def __post_init__(self): self.char_count = len(self.text)

@dataclass
class TextChunk:
    text: str
    source: str
    page: int
    chunk_id: int
    doc_type: str = "Document"

@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float
    chunk_id: int
    doc_type: str = "Document"


# ─── Multi-Format Extractors ──────────────────────────────────────────────────

def validate_file(filepath: str) -> Tuple[bool, str]:
    """Generic file validator for PDF, CSV, Excel, TXT, and Markdown formats."""
    path = Path(filepath)
    if not path.exists():        return False, "File does not exist."
    if path.stat().st_size == 0: return False, "File is empty."
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            with open(filepath, "rb") as f:
                if f.read(5) != b"%PDF-": return False, "Invalid PDF header."
        except Exception as e:
            return False, f"Cannot read PDF: {e}"
        return True, ""
    elif ext in (".csv", ".xlsx", ".xls", ".txt", ".md", ".json"):
        return True, ""
    return False, f"Unsupported file type: {ext}. Supported: PDF, CSV, XLSX, XLS, TXT, MD."


def extract_text_from_pdf(filepath: str) -> List[PageDocument]:
    """Layout-aware spatial block sorting for PDFs."""
    pages = []
    filename = Path(filepath).name
    try:
        doc = fitz.open(filepath)
    except Exception as e:
        print(f"  ❌ Cannot open PDF {filename}: {e}")
        return pages

    for idx in range(len(doc)):
        try:
            page = doc[idx]
            blocks = page.get_text("blocks")
            sorted_blocks = sorted(blocks, key=lambda b: (round(b[1] / 15) * 15, b[0]))
            
            block_texts = []
            for b in sorted_blocks:
                if b[6] == 0:  # Text block
                    t = b[4].strip()
                    if t:
                        block_texts.append(t)
            
            page_text = "\n\n".join(block_texts)
            if not page_text.strip():
                page_text = page.get_text("text")

            if len(page_text.strip()) >= 20:
                pages.append(PageDocument(text=page_text, source=filename, page=idx + 1, doc_type="PDF"))
        except Exception as e:
            print(f"  ⚠️ Error extracting page {idx+1} of {filename}: {e}")
            continue

    doc.close()
    return pages


def extract_text_from_csv(filepath: str) -> List[PageDocument]:
    """Convert CSV rows into PageDocuments grouped into 50-row chunks."""
    if not HAS_PANDAS:
        print("  ⚠️ pandas not installed — cannot parse CSV.")
        return []
    filename = Path(filepath).name
    pages = []
    try:
        df = pd.read_csv(filepath)
        ROWS_PER_PAGE = 50
        col_names = list(df.columns)
        row_texts = []
        for _, row in df.iterrows():
            parts = [f"{col}: {val}" for col, val in zip(col_names, row.values) if str(val).strip() not in ("", "nan", "None")]
            row_texts.append(" | ".join(parts))

        for page_num, start in enumerate(range(0, len(row_texts), ROWS_PER_PAGE), 1):
            block = "\n".join(row_texts[start:start + ROWS_PER_PAGE])
            if len(block.strip()) >= 20:
                pages.append(PageDocument(
                    text=f"[CSV Table: {filename}]\nColumns: {', '.join(str(c) for c in col_names)}\n\n{block}",
                    source=filename, page=page_num, doc_type="CSV"
                ))
        print(f"  ✅ CSV: {len(df)} rows → {len(pages)} page chunks")
    except Exception as e:
        print(f"  ❌ CSV parse error for {filename}: {e}")
    return pages


def extract_text_from_excel(filepath: str) -> List[PageDocument]:
    """Convert Excel workbook sheets into PageDocuments."""
    if not HAS_PANDAS:
        print("  ⚠️ pandas/openpyxl not installed — cannot parse Excel.")
        return []
    filename = Path(filepath).name
    pages = []
    try:
        xls = pd.ExcelFile(filepath)
        ROWS_PER_PAGE = 50
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            col_names = list(df.columns)
            row_texts = []
            for _, row in df.iterrows():
                parts = [f"{col}: {val}" for col, val in zip(col_names, row.values) if str(val).strip() not in ("", "nan", "None")]
                row_texts.append(" | ".join(parts))
            for page_num, start in enumerate(range(0, len(row_texts), ROWS_PER_PAGE), 1):
                block = "\n".join(row_texts[start:start + ROWS_PER_PAGE])
                if len(block.strip()) >= 20:
                    pages.append(PageDocument(
                        text=f"[Excel: {filename} | Sheet: {sheet_name}]\nColumns: {', '.join(str(c) for c in col_names)}\n\n{block}",
                        source=f"{filename}#{sheet_name}", page=page_num, doc_type="Excel"
                    ))
        print(f"  ✅ Excel: {len(xls.sheet_names)} sheet(s) → {len(pages)} page chunks")
    except Exception as e:
        print(f"  ❌ Excel parse error for {filename}: {e}")
    return pages


def extract_text_from_txt(filepath: str) -> List[PageDocument]:
    """Read plain text or markdown files."""
    filename = Path(filepath).name
    ext = Path(filepath).suffix.lower()
    doc_type = "Markdown" if ext == ".md" else "Text"
    pages = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        if len(content) >= 20:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200, chunk_overlap=150,
                separators=["\n## ", "\n### ", "\n\n", "\n", " "]
            )
            subs = splitter.split_text(content)
            for page_num, sub in enumerate(subs, 1):
                pages.append(PageDocument(text=sub, source=filename, page=page_num, doc_type=doc_type))
        print(f"  ✅ {doc_type}: {len(pages)} sections extracted from {filename}")
    except Exception as e:
        print(f"  ❌ Text parse error for {filename}: {e}")
    return pages


def scrape_website_urls(urls_text: str) -> Tuple[List[PageDocument], List[str]]:
    """Scrape and clean multiple website URLs with robust browser headers and SSL fallback."""
    if not HAS_BS4:
        return [], ["  ⚠️ `requests` or `beautifulsoup4` not installed in runtime environment."]
    
    raw_urls = [u.strip() for u in re.split(r"[\n,;]+", urls_text) if u.strip()]
    pages = []
    logs = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"macOS"',
        "Upgrade-Insecure-Requests": "1",
    }
    
    for raw_url in raw_urls:
        url = raw_url
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url
        try:
            try:
                resp = requests.get(url, headers=headers, timeout=12)
                resp.raise_for_status()
            except (requests.exceptions.SSLError, requests.exceptions.HTTPError):
                resp = requests.get(url, headers=headers, timeout=12, verify=False)
                resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "lxml")

            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form"]):
                tag.decompose()

            raw_text = soup.get_text(separator="\n", strip=True)
            raw_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            subs = splitter.split_text(raw_text)
            domain = url.split("//")[-1].split("/")[0]
            scraped_count = 0
            for page_num, sub in enumerate(subs, 1):
                if len(sub.strip()) >= 25:
                    pages.append(PageDocument(text=sub, source=f"web:{domain}", page=page_num, doc_type="Website"))
                    scraped_count += 1

            if scraped_count > 0:
                logs.append(f"  🌐 Scraped {scraped_count} sections from `{domain}` ({url})")
                print(f"  ✅ URL scraped: {scraped_count} chunks from {url}")
            else:
                logs.append(f"  ⚠️ No readable text found on `{url}`")
        except Exception as e:
            err_msg = str(e)
            if "Max retries exceeded" in err_msg or "Failed to establish" in err_msg:
                err_msg = "Connection timeout or domain unreachable"
            elif "403" in err_msg:
                err_msg = "HTTP 403 Forbidden (Blocked by website firewall)"
            logs.append(f"  ❌ Failed to scrape `{url}`: {err_msg}")
            print(f"  ❌ URL scrape error for {url}: {e}")

    return pages, logs


def extract_all_documents(file_paths: List[str]) -> List[PageDocument]:
    """Route each file to the appropriate extractor."""
    all_pages = []
    for fp in file_paths:
        ext = Path(fp).suffix.lower()
        if ext == ".pdf":
            all_pages.extend(extract_text_from_pdf(fp))
        elif ext == ".csv":
            all_pages.extend(extract_text_from_csv(fp))
        elif ext in (".xlsx", ".xls"):
            all_pages.extend(extract_text_from_excel(fp))
        elif ext in (".txt", ".md", ".json"):
            all_pages.extend(extract_text_from_txt(fp))
    return all_pages


# ─── Cleaning & Chunking ──────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    t = re.sub(r"([a-z])-\n([a-z])", r"\1\2", t, flags=re.IGNORECASE)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def clean_all_pages(pages: List[PageDocument]) -> List[PageDocument]:
    out = []
    for p in pages:
        c = clean_text(p.text)
        if len(c) >= 20:
            out.append(PageDocument(text=c, source=p.source, page=p.page, doc_type=p.doc_type))
    return out


def get_adaptive_chunk_config(doc_type: str, text_len: int) -> Tuple[int, int, List[str]]:
    """
    Dynamically computes optimal chunk_size, chunk_overlap, and splitting separators
    based on document format and content density.
    """
    doc_type_lower = (doc_type or "").lower()
    
    if doc_type_lower in ("csv", "excel", "table"):
        # Structured tabular records: chunk by whole rows, no destructive overlap
        chunk_size = 800 if text_len > 2000 else 500
        chunk_overlap = 0
        separators = ["\n", "\n\n", " | ", " "]
    elif doc_type_lower in ("markdown", "md"):
        # Markdown: split along header hierarchies
        chunk_size = 900 if text_len > 4000 else 650
        chunk_overlap = 100
        separators = ["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "]
    elif doc_type_lower in ("website", "web", "html"):
        # Scraped web content: adaptive to article length
        chunk_size = 850 if text_len > 5000 else 650
        chunk_overlap = 90
        separators = ["\n\n", "\n", ". ", "? ", "! ", " "]
    elif doc_type_lower in ("pdf",):
        # PDFs: section/paragraph aware
        chunk_size = 750 if text_len > 3000 else 550
        chunk_overlap = 100
        separators = ["\n\n", "\n", ". ", "? ", "! ", " "]
    else:
        # Default plain text
        chunk_size = 700
        chunk_overlap = 90
        separators = ["\n\n", "\n", ". ", " "]
        
    return chunk_size, chunk_overlap, separators


def create_chunks(pages: List[PageDocument], start_id: int = 0) -> List[TextChunk]:
    """
    Auto-adaptive multi-format chunker. Dynamically optimizes chunk boundaries,
    token limits, and semantic separators per document type (PDF, Web, CSV, Excel, MD).
    """
    chunks = []
    cid = start_id
    
    for p in pages:
        text = p.text.strip()
        if not text:
            continue
            
        doc_type = p.doc_type or "Text"
        text_len = len(text)
        
        # If content is short (<= 450 chars), preserve as a single unified chunk without fragmenting
        if text_len <= 450:
            if text_len >= 20:
                chunks.append(TextChunk(
                    text=text,
                    source=p.source,
                    page=p.page,
                    chunk_id=cid,
                    doc_type=doc_type
                ))
                cid += 1
            continue
            
        # Get adaptive parameters tailored to document structure
        chunk_size, chunk_overlap, separators = get_adaptive_chunk_config(doc_type, text_len)
        
        # Tabular data handling (CSV/Excel) - preserve row boundaries and schema prefix
        if doc_type in ("CSV", "Excel") and "\n" in text:
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            header_prefix = ""
            row_lines = []
            
            for line in lines:
                if line.startswith("[") or line.startswith("Columns:"):
                    header_prefix += line + "\n"
                else:
                    row_lines.append(line)
                    
            if row_lines:
                # Group rows dynamically into self-contained batches
                current_batch = []
                current_len = len(header_prefix)
                
                for row_str in row_lines:
                    row_len = len(row_str) + 1
                    if current_batch and (current_len + row_len > chunk_size):
                        chunk_body = (header_prefix.strip() + "\n\n" + "\n".join(current_batch)).strip()
                        if len(chunk_body) >= 25:
                            chunks.append(TextChunk(
                                text=chunk_body,
                                source=p.source,
                                page=p.page,
                                chunk_id=cid,
                                doc_type=doc_type
                            ))
                            cid += 1
                        current_batch = [row_str]
                        current_len = len(header_prefix) + row_len
                    else:
                        current_batch.append(row_str)
                        current_len += row_len
                        
                if current_batch:
                    chunk_body = (header_prefix.strip() + "\n\n" + "\n".join(current_batch)).strip()
                    if len(chunk_body) >= 25:
                        chunks.append(TextChunk(
                            text=chunk_body,
                            source=p.source,
                            page=p.page,
                            chunk_id=cid,
                            doc_type=doc_type
                        ))
                        cid += 1
                continue
                
        # Standard recursive text splitting for prose/web/PDF/markdown
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
        )
        
        sub_texts = splitter.split_text(text)
        for t in sub_texts:
            t = t.strip()
            if len(t) >= 25:
                chunks.append(TextChunk(
                    text=t,
                    source=p.source,
                    page=p.page,
                    chunk_id=cid,
                    doc_type=doc_type
                ))
                cid += 1
                
    return chunks


# ─── Embeddings & FAISS Index ─────────────────────────────────────────────────

_embedding_model: Optional[SentenceTransformer] = None

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        name = CONFIG["embedding_model"]
        print(f"  ⏳ Loading embedding model: {name}...")
        _embedding_model = SentenceTransformer(name)
        print(f"  ✅ Embedding model ready (dim={_embedding_model.get_sentence_embedding_dimension()})")
    return _embedding_model


def _base_embed_texts(texts: List[str], show_progress: bool = False) -> np.ndarray:
    model = get_embedding_model()
    embs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )
    return np.ascontiguousarray(embs, dtype=np.float32)

if HAS_SPACES:
    embed_texts = spaces.GPU(_base_embed_texts)
else:
    embed_texts = _base_embed_texts


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_knowledge_base(index: faiss.IndexFlatIP, chunks: List[TextChunk], manifest: Dict):
    try:
        faiss.write_index(index, CONFIG["faiss_index_path"])
        with open(CONFIG["chunks_path"], "wb") as f:
            pickle.dump(chunks, f)
        with open(CONFIG["manifest_path"], "wb") as f:
            pickle.dump(manifest, f)
        print(f"  💾 Knowledge base persisted ({index.ntotal} vectors, {len(chunks)} chunks)")
    except Exception as e:
        print(f"  ⚠️ Could not save knowledge base: {e}")


def load_knowledge_base() -> Tuple[Optional[faiss.IndexFlatIP], List[TextChunk], Dict]:
    ipath = CONFIG["faiss_index_path"]
    cpath = CONFIG["chunks_path"]
    mpath = CONFIG["manifest_path"]
    if Path(ipath).exists() and Path(cpath).exists():
        try:
            index = faiss.read_index(ipath)
            with open(cpath, "rb") as f:
                chunks = pickle.load(f)
            manifest = {}
            if Path(mpath).exists():
                with open(mpath, "rb") as f:
                    manifest = pickle.load(f)
            return index, chunks, manifest
        except Exception as e:
            print(f"  ⚠️ Could not load saved knowledge base: {e}")
    return None, [], {}


# ─── Advanced 2026 Multi-Stage Hybrid Retrieval ───────────────────────────────

def rewrite_query(question: str) -> str:
    """Adaptive query vocabulary expander."""
    q = question.strip()
    expansions = {
        "who is":       "name candidate author profile background summary role title",
        "education":    "education degree university college btech gpa qualification study",
        "skills":       "skills tools technologies programming python machine learning stack",
        "experience":   "experience work history internships employment company role",
        "projects":     "projects work portfolio achievements applications architecture",
        "contact":      "email phone linkedin location address contact github",
        "revenue":      "revenue profit sales financials balance sheet growth quarterly",
        "what is":      "definition overview introduction key components architecture",
    }
    extras = []
    q_lower = q.lower()
    for trigger, exp in expansions.items():
        if trigger in q_lower:
            extras.append(exp)
    return f"{q} {' '.join(extras)}" if extras else q


def retrieve_dense(query: str, index: faiss.IndexFlatIP, chunks: List[TextChunk], top_k: int = 10) -> List[RetrievedChunk]:
    if index is None or index.ntotal == 0 or not chunks:
        return []
    q_emb = embed_texts([query])
    k = min(top_k, index.ntotal)
    scores, indices = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(chunks):
            c = chunks[idx]
            results.append(RetrievedChunk(
                text=c.text, source=c.source, page=c.page,
                score=float(score), chunk_id=c.chunk_id, doc_type=c.doc_type
            ))
    return results


def retrieve_bm25(query: str, chunks: List[TextChunk], top_k: int = 10) -> List[Tuple[int, float]]:
    """BM25 sparse retrieval."""
    if not HAS_BM25 or not chunks:
        return []
    tokenized_corpus = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


def deduplicate_chunks(chunks: List[RetrievedChunk], cutoff: float = 0.95) -> List[RetrievedChunk]:
    if len(chunks) <= 1:
        return chunks
    embs = embed_texts([c.text for c in chunks])
    remove = set()
    for i in range(len(chunks)):
        if i in remove: continue
        for j in range(i + 1, len(chunks)):
            if j not in remove and float(np.dot(embs[i], embs[j])) >= cutoff:
                remove.add(j)
    return [chunks[i] for i in range(len(chunks)) if i not in remove]


def reorder_lost_in_the_middle(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    Reorders retrieved evidence so highest-scoring chunks are placed at the
    extremities (start and end) to prevent LLM attention loss in the middle.
    """
    if len(chunks) <= 2:
        return chunks
    reordered = []
    left = True
    for c in chunks:
        if left:
            reordered.append(c)
        else:
            reordered.insert(0, c)
        left = not left
    return reordered


def compute_adaptive_top_k(question: str, total_chunks: int) -> int:
    """
    Dynamically computes optimal top_k retrieved chunks based on knowledge base size
    and query intent (broad summary vs specific fact lookup).
    """
    if total_chunks <= 0:
        return 0
    if total_chunks <= 3:
        return total_chunks
        
    # Check if query seeks a comprehensive summary or comparison
    q_lower = (question or "").lower()
    is_broad_query = any(w in q_lower for w in ["summary", "summarize", "overview", "all", "list", "compare", "everything", "difference"])
    
    if is_broad_query:
        base_k = min(8, max(4, int(np.ceil(np.sqrt(total_chunks) * 1.5))))
    else:
        base_k = min(6, max(3, int(np.ceil(np.log2(total_chunks) * 1.5))))
        
    return min(base_k, total_chunks)


def hybrid_retrieve(question: str, index, chunks: List[TextChunk], top_k: int = None) -> List[RetrievedChunk]:
    """
    Combines FAISS Dense + BM25 Sparse with Reciprocal Rank Fusion (RRF)
    and Lost-in-the-Middle Context Reordering with auto-adaptive top-k.
    """
    if top_k is None:
        top_k = compute_adaptive_top_k(question, len(chunks))
        
    expanded_query = rewrite_query(question)

    dense_candidates = retrieve_dense(expanded_query, index, chunks, top_k=max(4, top_k * 2))
    dense_rank = {r.chunk_id: rank + 1 for rank, r in enumerate(dense_candidates)}

    if not HAS_BM25:
        deduped = deduplicate_chunks(dense_candidates[:top_k])
        return reorder_lost_in_the_middle(deduped)

    bm25_pairs = retrieve_bm25(expanded_query, chunks, top_k=max(4, top_k * 2))
    bm25_rank = {chunks[i].chunk_id: rank + 1 for rank, (i, _) in enumerate(bm25_pairs)}

    # Reciprocal Rank Fusion (k=60)
    k_rrf = 60
    all_ids = set(dense_rank.keys()) | set(bm25_rank.keys())
    rrf_scores: Dict[int, float] = {}
    for cid in all_ids:
        score = 0.0
        if cid in dense_rank: score += 1.0 / (k_rrf + dense_rank[cid])
        if cid in bm25_rank:  score += 1.0 / (k_rrf + bm25_rank[cid])
        rrf_scores[cid] = score

    chunk_by_id = {c.chunk_id: c for c in chunks}
    dense_score_by_id = {r.chunk_id: r.score for r in dense_candidates}

    merged_ids = sorted(all_ids, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
    results = []
    for cid in merged_ids:
        if cid in chunk_by_id:
            c = chunk_by_id[cid]
            results.append(RetrievedChunk(
                text=c.text, source=c.source, page=c.page,
                score=dense_score_by_id.get(cid, rrf_scores[cid]),
                chunk_id=c.chunk_id, doc_type=c.doc_type
            ))
    
    deduped = deduplicate_chunks(results)
    return reorder_lost_in_the_middle(deduped)


def enhanced_retrieve(question: str, index, chunks, top_k=None) -> List[RetrievedChunk]:
    return hybrid_retrieve(question, index, chunks, top_k=top_k)


# ─── Prompt Engineering ────────────────────────────────────────────────────────

def build_messages(question: str,
                   retrieved: List[RetrievedChunk],
                   history: List[Dict] = None) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    sep = "=" * 55
    if retrieved:
        context_parts = [f"{sep}\nDOCUMENT & WEB EVIDENCE:\n{sep}"]
        for i, chunk in enumerate(retrieved, 1):
            type_icon = {"PDF": "📄", "CSV": "📊", "Excel": "📈", "Website": "🌐", "Markdown": "📝", "Text": "📄"}.get(chunk.doc_type, "📁")
            context_parts.append(
                f"\n[Evidence {i}] {type_icon} Source: {chunk.source} (Page/Section {chunk.page}) | Relevance: {chunk.score:.2f}\n"
                f"{chunk.text}"
            )
        context_block = "\n".join(context_parts)
    else:
        context_block = "[No relevant context found in uploaded documents or web sources.]"

    if history:
        slice_ = history[-(CONFIG["max_history_turns"] * 2):]
        for turn in slice_:
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user",
        "content": f"{context_block}\n\n{sep}\nUSER QUESTION:\n{sep}\n{question}"
    })

    return messages


# ─── LLM Client (NVIDIA NIM) ───────────────────────────────────────────────────

_openai_client: Optional[OpenAI] = None

def configure_llm(api_key: str) -> Tuple[bool, str]:
    global _openai_client
    if not api_key or not api_key.strip():
        _openai_client = None
        return False, "⚠️ API Key is empty."
    try:
        _openai_client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=api_key.strip(),
        )
        return True, "✅ Connected to NVIDIA NIM API"
    except Exception as e:
        _openai_client = None
        return False, f"❌ Failed to connect: {e}"


def get_client() -> Optional[OpenAI]:
    return _openai_client


def stream_generate_answer(question: str,
                           retrieved: List[RetrievedChunk],
                           history: List[Dict] = None) -> Generator[str, None, None]:
    client = get_client()
    if client is None:
        yield "⚠️ **LLM Not Configured:** Please enter your NVIDIA API key in the **Setup** tab."
        return

    messages = build_messages(question, retrieved, history)
    model = CONFIG["llm_model"]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=CONFIG["llm_temperature"],
            top_p=CONFIG["llm_top_p"],
            max_tokens=CONFIG["llm_max_tokens"],
            stream=True,
        )

        full_text = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content if chunk.choices else ""
            if delta:
                full_text += delta
                yield full_text

    except Exception as e:
        err_msg = f"❌ **Inference Error ({model}):** {type(e).__name__}: {e}"
        yield err_msg


def format_sources_md(sources: List[Dict]) -> str:
    if not sources:
        return ""
    lines = ["📌 **Sources & Grounding:**"]
    for s in sources:
        icon = {"PDF": "📄", "CSV": "📊", "Excel": "📈", "Website": "🌐"}.get(s.get("type", ""), "📁")
        lines.append(
            f"- {icon} `{s['source']}` — Page/Section {s['page']} *(similarity: {s['score']:.2f})*"
        )
    return "\n".join(lines)


# ─── Application State ─────────────────────────────────────────────────────────

app_state: Dict[str, Any] = {
    "index":    None,
    "chunks":   [],
    "manifest": {},
    "history":  [],
}




def render_manifest_table_html(manifest: Dict) -> str:
    if not manifest:
        return """
        <div class="eval-placeholder">
            <div class="placeholder-icon">🗂️</div>
            <div class="placeholder-title">Knowledge Base is Empty</div>
            <div class="placeholder-desc">Upload documents or scrape website URLs above and click <b>Ingest & Index All Sources</b>.</div>
        </div>
        """
    
    total_chunks = sum(m.get("chunks", 0) for m in manifest.values())
    total_pages  = sum(m.get("pages", 0) for m in manifest.values())
    total_sources = len(manifest)

    html = [f'''
    <div class="custom-table-card">
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 14px 18px; border-bottom: 1px solid var(--card-border); background: var(--input-bg);">
            <div style="display: flex; gap: 10px; align-items: center;">
                <span style="font-weight: 700; color: var(--text-primary); font-size: 0.95rem;">📚 Active Knowledge Inventory</span>
                <span class="badge badge-cyan">{total_sources} Source{'s' if total_sources != 1 else ''}</span>
                <span class="badge badge-purple">{total_chunks} Total Chunks</span>
            </div>
            <div style="font-size: 0.8rem; color: var(--text-secondary);">
                Hybrid Search: <b style="color: var(--accent-green);">Active (FAISS + BM25)</b>
            </div>
        </div>
        <table class="custom-eval-table">
            <thead>
                <tr>
                    <th style="width: 35%; text-align: left;">Source Name / URL</th>
                    <th style="width: 18%; text-align: center;">Format</th>
                    <th style="width: 15%; text-align: center;">Pages / Sections</th>
                    <th style="width: 15%; text-align: center;">Chunks</th>
                    <th style="width: 17%; text-align: center;">Index Status</th>
                </tr>
            </thead>
            <tbody>
    ''']

    for name, meta in manifest.items():
        doc_type = meta.get("type", "Document")
        pages = meta.get("pages", 1)
        chunks = meta.get("chunks", 1)
        date_str = meta.get("date", "Active")
        
        type_badge_class = {
            "PDF": "badge-blue", "CSV": "badge-green",
            "Excel": "badge-green", "Website": "badge-purple",
            "Markdown": "badge-cyan", "Text": "badge-slate"
        }.get(doc_type, "badge-slate")

        icon = {"PDF": "📄", "CSV": "📊", "Excel": "📈", "Website": "🌐", "Markdown": "📝", "Text": "📄"}.get(doc_type, "📁")

        html.append(f'''
        <tr>
            <td style="font-weight: 600; color: var(--text-primary);">{icon} {name}</td>
            <td style="text-align: center;"><span class="badge {type_badge_class}">{doc_type}</span></td>
            <td style="text-align: center; color: var(--text-secondary);">{pages}</td>
            <td style="text-align: center; color: var(--accent-blue); font-weight: 600;">{chunks}</td>
            <td style="text-align: center;"><span class="badge badge-green">🟢 Ready ({date_str})</span></td>
        </tr>
        ''')

    html.append('</tbody></table></div>')
    return "".join(html)


# ─── Unified Ingestion Pipeline ───────────────────────────────────────────────

def cb_unified_ingest(files, urls_text: str, append_mode: bool):
    """
    Ingests files (PDF, CSV, Excel, TXT, MD) AND Website URLs simultaneously
    with support for either Fresh Rebuild or Incremental Append.
    """
    start_time = time.time()
    log_lines = [f"🚀 **[2026 RAG Ingestion Pipeline Started at {datetime.now().strftime('%H:%M:%S')}]**\n"]
    
    new_pages: List[PageDocument] = []
    new_manifest_entries: Dict[str, Dict] = {}
    today_time = datetime.now().strftime("%H:%M")

    used_sources = set(app_state.get("manifest", {}).keys()) if append_mode else set()

    # 1. Process Uploaded Files
    if files:
        for f in files:
            fp = f.name
            ok, err = validate_file(fp)
            raw_fname = Path(fp).name
            
            # Ensure unique distinct source name even if user uploads multiple files with identical basenames
            unique_fname = raw_fname
            counter = 2
            while unique_fname in used_sources or unique_fname in new_manifest_entries:
                stem = Path(raw_fname).stem
                suffix = Path(raw_fname).suffix
                unique_fname = f"{stem} ({counter}){suffix}"
                counter += 1
            used_sources.add(unique_fname)

            if ok:
                doc_pages = extract_all_documents([fp])
                if doc_pages:
                    for p in doc_pages:
                        p.source = unique_fname
                    new_pages.extend(doc_pages)
                    doc_type = doc_pages[0].doc_type
                    new_manifest_entries[unique_fname] = {
                        "type": doc_type,
                        "pages": len(doc_pages),
                        "chunks": 0,  # Updated after chunking
                        "date": today_time,
                    }
                    log_lines.append(f"  ✅ Extracted {len(doc_pages)} page(s) from `{unique_fname}` ({doc_type})")
                else:
                    log_lines.append(f"  ⚠️ No readable text in `{raw_fname}`")
            else:
                log_lines.append(f"  ❌ Invalid file `{raw_fname}`: {err}")

    # 2. Process Website URLs
    if urls_text and urls_text.strip():
        web_pages, web_logs = scrape_website_urls(urls_text)
        for wl in web_logs:
            log_lines.append(wl)

        if web_pages:
            new_pages.extend(web_pages)
            # Group by domain
            domains = set(p.source for p in web_pages)
            for d in domains:
                d_pages = [p for p in web_pages if p.source == d]
                new_manifest_entries[d] = {
                    "type": "Website",
                    "pages": len(d_pages),
                    "chunks": 0,
                    "date": today_time,
                }

    if not new_pages:
        log_lines.append("\n⚠️ **No valid documents or web content found to index.**")
        current_manifest_html = render_manifest_table_html(app_state["manifest"])
        return "\n".join(log_lines), current_manifest_html, gr.update(visible=False)

    # 3. Clean & Chunk
    cleaned_pages = clean_all_pages(new_pages)
    
    if append_mode and app_state["index"] is not None and len(app_state["chunks"]) > 0:
        start_chunk_id = len(app_state["chunks"])
        new_chunks = create_chunks(cleaned_pages, start_id=start_chunk_id)
        if not new_chunks:
            log_lines.append("❌ Chunks could not be created.")
            return "\n".join(log_lines), render_manifest_table_html(app_state["manifest"]), gr.update(visible=False)

        log_lines.append(f"\n⏳ Embedding {len(new_chunks)} new chunks (all-MiniLM-L6-v2)...")
        new_embeddings = embed_texts([c.text for c in new_chunks], show_progress=False)
        app_state["index"].add(new_embeddings)
        app_state["chunks"].extend(new_chunks)
        
        # Update manifest chunks count
        for c in new_chunks:
            if c.source in new_manifest_entries:
                new_manifest_entries[c.source]["chunks"] = new_manifest_entries[c.source].get("chunks", 0) + 1
        app_state["manifest"].update(new_manifest_entries)
        mode_desc = f"Appended {len(new_chunks)} chunks to active index"

    else:
        # Fresh Rebuild
        new_chunks = create_chunks(cleaned_pages, start_id=0)
        if not new_chunks:
            log_lines.append("❌ Chunks could not be created.")
            return "\n".join(log_lines), render_manifest_table_html(app_state["manifest"]), gr.update(visible=False)

        log_lines.append(f"\n⏳ Embedding {len(new_chunks)} total chunks (all-MiniLM-L6-v2)...")
        embeddings = embed_texts([c.text for c in new_chunks], show_progress=False)
        app_state["index"] = build_faiss_index(embeddings)
        app_state["chunks"] = new_chunks
        app_state["history"] = []
        
        for c in new_chunks:
            if c.source in new_manifest_entries:
                new_manifest_entries[c.source]["chunks"] = new_manifest_entries[c.source].get("chunks", 0) + 1
        app_state["manifest"] = new_manifest_entries
        mode_desc = f"Freshly built index with {len(new_chunks)} chunks"

    # Persist
    save_knowledge_base(app_state["index"], app_state["chunks"], app_state["manifest"])
    
    elapsed = time.time() - start_time
    total_vectors = app_state["index"].ntotal
    total_sources = len(app_state["manifest"])

    log_lines.append(
        f"\n✨ **Knowledge Base Ready!**\n"
        f"   • Total Sources  : {total_sources} active sources (PDF/CSV/Excel/Web)\n"
        f"   • Total Vectors  : {total_vectors} vectors indexed in FAISS + BM25\n"
        f"   • Indexing Mode  : {mode_desc}\n"
        f"   • Total Time     : {elapsed:.2f}s\n\n"
        f"👉 **Ready!** Switch to the **💬 Chat with Documents** tab to ask cross-source questions!"
    )

    manifest_html = render_manifest_table_html(app_state["manifest"])
    return "\n".join(log_lines), manifest_html, gr.update(visible=True)



def cb_clear_knowledge_base():
    """Resets in-memory and on-disk knowledge base."""
    app_state["index"] = None
    app_state["chunks"] = []
    app_state["manifest"] = {}
    app_state["history"] = []
    for p in [CONFIG["faiss_index_path"], CONFIG["chunks_path"], CONFIG["manifest_path"]]:
        if os.path.exists(p):
            os.remove(p)
    empty_manifest_html = render_manifest_table_html({})
    return "\ud83d\uddc1\ufe0f Knowledge Base cleared. All vectors and sources removed.", empty_manifest_html, gr.update(visible=False)


# ─── Gradio Callbacks ──────────────────────────────────────────────────────────

def format_api_status_html(is_connected: bool, message: str = "", masked_key: str = "", model_id: str = "") -> str:
    if is_connected:
        return f'''
        <div class="status-badge-row">
            <span class="badge badge-green">✅ Connected to NVIDIA NIM API</span>
            <span class="badge badge-slate" style="font-family: 'JetBrains Mono', monospace;">{masked_key}</span>
        </div>
        '''
    else:
        return f'''
        <div class="status-badge-row">
            <span class="badge badge-red">⚠️ No API Key Configured</span>
            <span style="font-size: 0.82rem; color: #94a3b8;">{message}</span>
        </div>
        '''


def format_model_status_html(model_id: str) -> str:
    return f'''
    <div class="status-badge-row">
        <span style="font-size: 0.84rem; color: #94a3b8; font-weight: 600;">Active Model:</span>
        <span class="badge badge-cyan" style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;">{model_id}</span>
    </div>
    '''


def cb_configure_api(api_key: str, model_choice: str) -> str:
    api_key = api_key.strip()
    if not api_key:
        return format_api_status_html(False, "Please enter your NVIDIA API key below and click Connect.")
    selected_model_id = AVAILABLE_MODELS.get(model_choice, model_choice)
    CONFIG["llm_model"] = selected_model_id
    ok, msg = configure_llm(api_key)
    if ok:
        masked = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "Active"
        return format_api_status_html(True, masked_key=masked, model_id=selected_model_id)
    return format_api_status_html(False, message=f"Connection failed: {msg}")


def cb_update_model(model_choice: str) -> str:
    selected_model_id = AVAILABLE_MODELS.get(model_choice, model_choice)
    CONFIG["llm_model"] = selected_model_id
    return format_model_status_html(selected_model_id)


def cb_chat_stream(message: str, history: List[Dict[str, str]]):
    """Real-time streaming multi-source RAG conversation handler."""
    if not message.strip():
        yield history or [], ""
        return

    if history is None:
        history = []

    # Check if documents are ready
    if app_state["index"] is None or app_state["index"].ntotal == 0:
        reply = "⚠️ **Knowledge Base is Empty:** Please upload files (PDF/CSV/Excel) or scrape a Website URL in the **Upload & Ingest** tab first."
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply}
        ], ""
        return

    # Check if API is ready
    if get_client() is None:
        reply = "⚠️ **NVIDIA API Not Configured:** Please enter your API key in the **Setup & Model Selector** tab."
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply}
        ], ""
        return

    # Append user question and thinking indicator
    current_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "⚡ *Searching multi-source knowledge & reasoning...*"}
    ]
    yield current_history, ""

    try:
        t0 = time.time()
        retrieved = enhanced_retrieve(message, app_state["index"], app_state["chunks"])
        retrieval_time = time.time() - t0

        t1 = time.time()
        answer_stream = stream_generate_answer(message, retrieved, app_state["history"])

        # Format sources
        seen, sources = set(), []
        for chunk in retrieved:
            key = (chunk.source, chunk.page)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": chunk.source,
                    "page": chunk.page,
                    "score": chunk.score,
                    "type": chunk.doc_type
                })

        full_answer = ""
        for partial_answer in answer_stream:
            full_answer = partial_answer
            current_history[-1]["content"] = partial_answer + " ▌"
            yield current_history, ""

        gen_time = time.time() - t1
        model_name = CONFIG["llm_model"].split("/")[-1]

        final_content = (
            full_answer + "\n\n" +
            format_sources_md(sources) +
            f"\n\n*⏱️ Retrieval: {retrieval_time:.2f}s | Generation: {gen_time:.2f}s | Hybrid Mode: `BM25 + FAISS` | Model: `{model_name}`*"
        )

        current_history[-1]["content"] = final_content

        app_state["history"].append({"role": "user", "content": message})
        app_state["history"].append({"role": "assistant", "content": full_answer})
        max_h = CONFIG["max_history_turns"] * 2
        if len(app_state["history"]) > max_h:
            app_state["history"] = app_state["history"][-max_h:]

        yield current_history, ""

    except Exception as e:
        current_history[-1]["content"] = f"❌ **Error:** {type(e).__name__}: {e}"
        yield current_history, ""


def cb_clear_chat() -> List:
    app_state["history"] = []
    return []


# ─── Evaluation Handlers ──────────────────────────────────────────────────────

REFUSAL_KEYWORDS = [
    "not find", "cannot find", "not available", "not present", "no information",
    "unable to", "don't have", "not mentioned", "couldn't find", "not in the",
    "not discussed", "outside the scope",
]

def run_halluc_tests(questions_text: str):
    if app_state["index"] is None:
        return [["—", "—", "❌ No docs indexed. Please upload files in Tab 2.", "—"]]
    if get_client() is None:
        return [["—", "—", "❌ LLM not configured. Please enter API key in Tab 1.", "—"]]
    
    rows = []
    for q in questions_text.strip().split("\n"):
        q = q.strip()
        if not q: continue
        try:
            retrieved = enhanced_retrieve(q, app_state["index"], app_state["chunks"])
            ans_gen = stream_generate_answer(q, retrieved, [])
            ans = ""
            for p in ans_gen: ans = p
            refused = any(k in ans.lower() for k in REFUSAL_KEYWORDS)
            status = "✅ Refused (Grounded & Safe)" if refused else "ℹ️ Answered"
            rows.append([
                q,
                len(retrieved),
                status,
                ans[:140] + ("..." if len(ans) > 140 else "")
            ])
        except Exception as e:
            rows.append([q, 0, f"⚠️ Error: {type(e).__name__}", str(e)])
    return rows


def run_topk_exp(question: str):
    if app_state["index"] is None:
        return [["—"] * 5]
    rows = []
    for k in [3, 5, 8]:
        try:
            t0 = time.time()
            retrieved = retrieve_dense(question, app_state["index"], app_state["chunks"], top_k=k)
            r_time = time.time() - t0
            ans_gen = stream_generate_answer(question, retrieved, []) if get_client() else None
            ans = ""
            if ans_gen:
                for p in ans_gen: ans = p
            avg_score = round(float(np.mean([c.score for c in retrieved])), 4) if retrieved else 0.0
            unique_pages = len(set((c.source, c.page) for c in retrieved))
            rows.append([k, len(retrieved), avg_score, unique_pages, len(ans)])
        except Exception as e:
            rows.append([k, 0, 0, 0, 0])
    return rows


def render_hallucination_table_html(rows: List[List]) -> str:
    if not rows:
        return """
        <div class="eval-placeholder">
            <div class="placeholder-icon">🛡️</div>
            <div class="placeholder-title">Benchmark Not Executed Yet</div>
            <div class="placeholder-desc">Click <b>Run Hallucination Benchmark</b> above to test zero-hallucination guardrails across out-of-domain queries.</div>
        </div>
        """
    
    html = ['<div class="custom-table-card"><table class="custom-eval-table"><thead><tr>']
    html.append('<th style="width: 32%; text-align: left;">Test Question</th>')
    html.append('<th style="width: 14%; text-align: center;">Chunks Found</th>')
    html.append('<th style="width: 22%; text-align: center;">Safety Guardrail Status</th>')
    html.append('<th style="width: 32%; text-align: left;">Grounded Answer Output</th>')
    html.append('</tr></thead><tbody>')
    
    for r in rows:
        q = str(r[0]) if len(r) > 0 else ""
        chunks = str(r[1]) if len(r) > 1 else "0"
        status = str(r[2]) if len(r) > 2 else ""
        preview = str(r[3]) if len(r) > 3 else ""
        
        if "Refused" in status or "✅" in status:
            status_badge = '<span class="badge badge-green">🛡️ Refused (Grounded & Safe)</span>'
        elif "Error" in status or "❌" in status or "⚠️" in status:
            status_badge = f'<span class="badge badge-red">{status}</span>'
        else:
            status_badge = '<span class="badge badge-blue">ℹ️ Answered</span>'
            
        chunks_badge = f'<span class="badge badge-cyan">{chunks} chunks</span>' if chunks not in ["—", "0"] else '<span class="badge badge-slate">0 chunks</span>'
        
        html.append(f'''
        <tr>
            <td class="q-cell">{q}</td>
            <td style="text-align: center;">{chunks_badge}</td>
            <td style="text-align: center;">{status_badge}</td>
            <td class="preview-cell">{preview}</td>
        </tr>
        ''')
        
    html.append('</tbody></table></div>')
    return "".join(html)


def render_topk_table_html(rows: List[List]) -> str:
    if not rows:
        return """
        <div class="eval-placeholder">
            <div class="placeholder-icon">🔬</div>
            <div class="placeholder-title">Study Not Run Yet</div>
            <div class="placeholder-desc">Click <b>Run top_k Study</b> above to analyze density, similarity distributions, and answer depth.</div>
        </div>
        """
        
    html = ['<div class="custom-table-card"><table class="custom-eval-table"><thead><tr>']
    html.append('<th style="width: 14%; text-align: center;">Top-k Tested</th>')
    html.append('<th style="width: 18%; text-align: center;">Chunks Retrieved</th>')
    html.append('<th style="width: 22%; text-align: center;">Avg Cosine Similarity</th>')
    html.append('<th style="width: 20%; text-align: center;">Unique Pages/Sources</th>')
    html.append('<th style="width: 26%; text-align: center;">Answer Character Length</th>')
    html.append('</tr></thead><tbody>')
    
    for r in rows:
        k = str(r[0]) if len(r) > 0 else ""
        chunks = str(r[1]) if len(r) > 1 else ""
        sim = str(r[2]) if len(r) > 2 else ""
        pages = str(r[3]) if len(r) > 3 else ""
        ans_len = str(r[4]) if len(r) > 4 else ""
        
        html.append(f'''
        <tr>
            <td style="text-align: center;"><span class="badge badge-purple">k = {k}</span></td>
            <td style="text-align: center;"><span class="badge badge-cyan">{chunks} chunks</span></td>
            <td style="text-align: center; font-weight: 700; color: #34d399;">{sim}</td>
            <td style="text-align: center;"><span class="badge badge-slate">{pages} pages</span></td>
            <td style="text-align: center; color: #cbd5e1;">{ans_len} chars</td>
        </tr>
        ''')
        
    html.append('</tbody></table></div>')
    return "".join(html)


def cb_run_halluc_eval(questions_text: str) -> str:
    rows = run_halluc_tests(questions_text)
    return render_hallucination_table_html(rows)


def cb_run_topk_eval(question: str) -> str:
    rows = run_topk_exp(question)
    return render_topk_table_html(rows)


# ─── UI & Styling (Apple-Grade Dark Glassmorphism) ────────────────────────────

CSS = """
/* ─── CSS Variables (Dynamic Theme System — Default: White / Light) ────────── */
:root {
    --bg-base:                #f8fafc;
    --bg-surface:             #ffffff;
    --bg-card:                #ffffff;
    --bg-card-solid:          #ffffff;
    --bg-input:               #f8fafc;
    --border-subtle:          rgba(0, 0, 0, 0.10);
    --border-focus:           rgba(2, 132, 199, 0.6);
    --accent-blue:            #0284c7;
    --accent-green:           #059669;
    --accent-purple:          #7c3aed;
    --text-primary:           #0f172a;
    --text-secondary:         #334155;
    --text-muted:             #64748b;
    --radius-card:            20px;
    --radius-input:           12px;
    --radius-btn:             12px;
    
    /* Header Specific (Light Default) */
    --header-bg:              linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(241, 245, 249, 0.90) 100%);
    --header-border:          rgba(0, 0, 0, 0.08);
    --header-shadow:          0 8px 32px rgba(0, 0, 0, 0.06);
    --header-title-gradient:  linear-gradient(135deg, #0f172a 0%, #0284c7 60%, #4f46e5 100%);
    --toggle-btn-bg:          rgba(0, 0, 0, 0.06);
    --toggle-btn-border:      rgba(0, 0, 0, 0.12);
    --toggle-btn-color:       #0f172a;
    --toggle-btn-hover-bg:    rgba(0, 0, 0, 0.10);
    
    /* Card & Container Variables (Light Default) */
    --card-bg:                #ffffff;
    --card-border:            rgba(0, 0, 0, 0.10);
    --card-shadow:            0 8px 30px rgba(0, 0, 0, 0.06);
    
    /* Segmented Tab Bar Variables (Light Default) */
    --tab-nav-bg:             #ffffff;
    --tab-nav-border:         rgba(0, 0, 0, 0.12);
    --tab-nav-shadow:         0 4px 20px rgba(0, 0, 0, 0.08);
    --tab-btn-color:          #475569;
    --tab-btn-hover-bg:       #f1f5f9;
    --tab-btn-hover-color:    #0f172a;
    --tab-btn-active-bg:      linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
    --tab-btn-active-border:  rgba(2, 132, 199, 0.50);
    --tab-btn-active-shadow:  0 4px 14px rgba(2, 132, 199, 0.35);
    
    /* Form & Input Variables (Light Default) */
    --input-bg:               #f8fafc;
    --input-border:           rgba(0, 0, 0, 0.14);
    --input-color:            #0f172a;
    --input-shadow:           inset 0 1px 2px rgba(0, 0, 0, 0.04);
    --dropdown-menu-bg:       #ffffff;
    --dropdown-item-color:    #334155;
    --dropdown-item-hover:    #0284c7;
    --dropdown-item-hover-bg: #f0f9ff;
    --dropdown-item-sel-bg:   #e0f2fe;
    --dropdown-border:        rgba(0, 0, 0, 0.12);
    --dropdown-shadow:        0 16px 36px rgba(0, 0, 0, 0.12), 0 0 0 1px rgba(2, 132, 199, 0.20);
    
    /* Chat & Bot Bubbles (Light Default) */
    --bot-bubble-bg:          #f1f5f9;
    --bot-bubble-border:      rgba(0, 0, 0, 0.08);
    --bot-bubble-color:       #0f172a;
    
    /* Code & Tag Variables (Light Default) */
    --code-bg:                rgba(2, 132, 199, 0.10);
    --code-border:            rgba(2, 132, 199, 0.28);
    --code-color:             #0369a1;
}

/* ─── Dark Mode Variable Overrides ───────────────────────────────────────── */
:root.dark,
html.dark,
body.dark,
.gradio-container.dark,
.dark-theme,
body.dark-theme,
[data-theme="dark"],
body[data-theme="dark"] {
    --bg-base:                #050811;
    --bg-surface:             #0b1120;
    --bg-card:                rgba(15, 23, 42, 0.80);
    --bg-card-solid:          #0c1223;
    --bg-input:               #070d1a;
    --border-subtle:          rgba(255, 255, 255, 0.08);
    --border-focus:           rgba(56, 189, 248, 0.6);
    --accent-blue:            #38bdf8;
    --accent-green:           #34d399;
    --accent-purple:          #a78bfa;
    --text-primary:           #f8fafc;
    --text-secondary:         #94a3b8;
    --text-muted:             #64748b;
    
    /* Header Specific (Dark) */
    --header-bg:              linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.6) 100%);
    --header-border:          rgba(255, 255, 255, 0.09);
    --header-shadow:          0 8px 32px rgba(0, 0, 0, 0.45);
    --header-title-gradient:  linear-gradient(135deg, #ffffff 0%, #38bdf8 60%, #818cf8 100%);
    --toggle-btn-bg:          rgba(255, 255, 255, 0.08);
    --toggle-btn-border:      rgba(255, 255, 255, 0.16);
    --toggle-btn-color:       #e2e8f0;
    --toggle-btn-hover-bg:    rgba(255, 255, 255, 0.15);
    
    /* Card & Container Variables (Dark) */
    --card-bg:                rgba(12, 18, 35, 0.85);
    --card-border:            rgba(255, 255, 255, 0.09);
    --card-shadow:            0 10px 30px rgba(0, 0, 0, 0.45);
    
    /* Segmented Tab Bar Variables (Dark) */
    --tab-nav-bg:             rgba(11, 19, 43, 0.88);
    --tab-nav-border:         rgba(56, 189, 248, 0.25);
    --tab-nav-shadow:         0 8px 32px rgba(0, 0, 0, 0.50);
    --tab-btn-color:          #94a3b8;
    --tab-btn-hover-bg:       rgba(255, 255, 255, 0.08);
    --tab-btn-hover-color:    #f8fafc;
    --tab-btn-active-bg:      linear-gradient(135deg, rgba(37, 99, 235, 0.90) 0%, rgba(14, 165, 233, 0.95) 100%);
    --tab-btn-active-border:  rgba(255, 255, 255, 0.25);
    --tab-btn-active-shadow:  0 4px 18px rgba(14, 165, 233, 0.40);
    
    /* Form & Input Variables (Dark) */
    --input-bg:               #070d1a;
    --input-border:           rgba(255, 255, 255, 0.14);
    --input-color:            #f8fafc;
    --input-shadow:           inset 0 2px 4px rgba(0, 0, 0, 0.35);
    --dropdown-menu-bg:       #090e1a;
    --dropdown-item-color:    #e2e8f0;
    --dropdown-item-hover:    #38bdf8;
    --dropdown-item-hover-bg: rgba(56, 189, 248, 0.20);
    --dropdown-item-sel-bg:   rgba(56, 189, 248, 0.25);
    --dropdown-border:        rgba(56, 189, 248, 0.45);
    --dropdown-shadow:        0 24px 60px rgba(0, 0, 0, 0.95);
    
    /* Chat & Bot Bubbles (Dark) */
    --bot-bubble-bg:          rgba(15, 23, 42, 0.85);
    --bot-bubble-border:      rgba(255, 255, 255, 0.09);
    --bot-bubble-color:       #e2e8f0;
    
    /* Code & Tag Variables (Dark) */
    --code-bg:                rgba(56, 189, 248, 0.15);
    --code-border:            rgba(56, 189, 248, 0.35);
    --code-color:             #7dd3fc;
}

/* Global Markdown & Typography Rules */
h1, h2, h3, h4, h5, h6,
.prose h1, .prose h2, .prose h3, .prose h4,
.markdown h1, .markdown h2, .markdown h3, .markdown h4 {
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
}

p, span, label, li,
.prose p, .prose span, .prose li,
.markdown p, .markdown span, .markdown li {
    color: var(--text-secondary) !important;
}

strong, b,
.prose strong, .prose b,
.markdown strong, .markdown b {
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    font-weight: 700 !important;
}

body, .gradio-container {
    background-color: var(--bg-base) !important;
    background-image: 
        radial-gradient(ellipse at 15% 0%, rgba(2, 132, 199, 0.08) 0%, transparent 60%),
        radial-gradient(ellipse at 85% 10%, rgba(124, 58, 237, 0.06) 0%, transparent 55%),
        radial-gradient(ellipse at 50% 100%, rgba(5, 150, 105, 0.05) 0%, transparent 60%) !important;
    background-attachment: fixed !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Plus Jakarta Sans', 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
    transition: background-color 0.3s ease, color 0.3s ease;
}

body.dark, .gradio-container.dark, [data-theme="dark"], body.dark-theme {
    background-image: 
        radial-gradient(ellipse at 15% 0%, rgba(37, 99, 235, 0.12) 0%, transparent 60%),
        radial-gradient(ellipse at 85% 10%, rgba(139, 92, 246, 0.10) 0%, transparent 55%),
        radial-gradient(ellipse at 50% 100%, rgba(16, 185, 129, 0.06) 0%, transparent 60%) !important;
}

#header-container {
    background: var(--header-bg) !important;
    padding: 32px 28px 24px;
    border-radius: var(--radius-card);
    margin-bottom: 24px;
    border: 1px solid var(--header-border) !important;
    box-shadow: var(--header-shadow) !important;
    text-align: center;
    backdrop-filter: blur(24px);
    position: relative;
    transition: background 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
}

#header-container h1 {
    font-size: 2.35rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    margin: 0 0 8px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 12px !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: transparent !important;
    color: var(--text-primary) !important;
}

#header-container .header-logo-icon {
    font-size: 2.4rem !important;
    display: inline-block !important;
    filter: drop-shadow(0 4px 12px rgba(2, 132, 199, 0.25));
    animation: floatLogo 3s ease-in-out infinite alternate;
}

@keyframes floatLogo {
    0% { transform: translateY(0px) rotate(0deg); }
    100% { transform: translateY(-3px) rotate(3deg); }
}

#header-container .header-title-text {
    display: inline-block !important;
    font-weight: 800 !important;
    background: var(--header-title-gradient) !important;
    background-image: var(--header-title-gradient) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    color: var(--text-primary) !important;
    line-height: 1.2 !important;
}

body.dark #header-container .header-title-text,
.dark #header-container .header-title-text,
.dark-theme #header-container .header-title-text,
body.dark-theme #header-container .header-title-text,
[data-theme="dark"] #header-container .header-title-text,
body[data-theme="dark"] #header-container .header-title-text,
.gradio-container.dark #header-container .header-title-text {
    background: linear-gradient(135deg, #ffffff 0%, #38bdf8 55%, #a5b4fc 100%) !important;
    background-image: linear-gradient(135deg, #ffffff 0%, #38bdf8 55%, #a5b4fc 100%) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    color: #ffffff !important;
}

#header-container p {
    color: var(--text-secondary) !important;
    font-size: 0.96rem !important;
    font-weight: 500 !important;
    margin: 0 0 16px !important;
}

#theme-toggle-btn {
    position: absolute;
    top: 18px;
    right: 20px;
    background: var(--toggle-btn-bg) !important;
    border: 1px solid var(--toggle-btn-border) !important;
    color: var(--toggle-btn-color) !important;
    border-radius: 9999px;
    padding: 6px 14px;
    font-size: 0.80rem;
    font-weight: 600;
    cursor: pointer !important;
    pointer-events: auto !important;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s ease;
    backdrop-filter: blur(12px);
    z-index: 9999 !important;
}

#theme-toggle-btn * {
    pointer-events: none !important;
}

#theme-toggle-btn:hover {
    background: var(--toggle-btn-hover-bg) !important;
    border-color: rgba(56, 189, 248, 0.4) !important;
    transform: translateY(-1px);
}

.header-badges-row {
    display: flex !important;
    justify-content: center !important;
    gap: 10px !important;
    flex-wrap: wrap !important;
    margin-top: 4px !important;
}

.badge-tag {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    padding: 6px 16px !important;
    border-radius: 9999px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s ease !important;
    user-select: none !important;
}

.badge-tag:hover {
    transform: translateY(-1px) !important;
}

/* Light Mode Badges (Vivid, High Contrast, Modern) */
.badge-fast {
    background: rgba(245, 158, 11, 0.12) !important;
    border: 1px solid rgba(245, 158, 11, 0.40) !important;
    color: #b45309 !important;
}

.badge-search {
    background: rgba(2, 132, 199, 0.12) !important;
    border: 1px solid rgba(2, 132, 199, 0.40) !important;
    color: #0369a1 !important;
}

.badge-guard {
    background: rgba(99, 102, 241, 0.12) !important;
    border: 1px solid rgba(99, 102, 241, 0.40) !important;
    color: #4338ca !important;
}

.badge-ingest {
    background: rgba(219, 39, 119, 0.12) !important;
    border: 1px solid rgba(219, 39, 119, 0.40) !important;
    color: #be185d !important;
}

/* Dark Mode Badges */
body.dark .badge-fast, .dark .badge-fast, [data-theme="dark"] .badge-fast {
    background: rgba(245, 158, 11, 0.16) !important;
    border: 1px solid rgba(245, 158, 11, 0.45) !important;
    color: #fbbf24 !important;
}

body.dark .badge-search, .dark .badge-search, [data-theme="dark"] .badge-search {
    background: rgba(56, 189, 248, 0.16) !important;
    border: 1px solid rgba(56, 189, 248, 0.45) !important;
    color: #38bdf8 !important;
}

body.dark .badge-guard, .dark .badge-guard, [data-theme="dark"] .badge-guard {
    background: rgba(129, 140, 248, 0.16) !important;
    border: 1px solid rgba(129, 140, 248, 0.45) !important;
    color: #818cf8 !important;
}

body.dark .badge-ingest, .dark .badge-ingest, [data-theme="dark"] .badge-ingest {
    background: rgba(236, 72, 153, 0.16) !important;
    border: 1px solid rgba(236, 72, 153, 0.45) !important;
    color: #f472b6 !important;
}

/* ─── Modern Segmented Navigation Tabs ─────────────────────────────────── */
.tabs,
div[data-testid="tabs"],
div.tabs {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    width: 100% !important;
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Outer Tab Bar Centering Wrapper: Strict ZERO border, ZERO background, ZERO shadow */
.tabs > div:first-child,
div[data-testid="tabs"] > div:first-child,
div.tabs > div:first-child,
.tab-wrapper,
.tab-nav-wrapper {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    margin: 4px auto 26px auto !important;
    padding: 0 !important;
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    border-top: none !important;
    border-bottom: none !important;
    border-left: none !important;
    border-right: none !important;
    box-shadow: none !important;
    outline: none !important;
}

.tabitem,
div.tabitem,
.tabs > div.tabitem,
div[data-testid="tabs"] > div.tabitem {
    width: 100% !important;
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Remove all pseudo-element lines, underlines, and default Gradio dividers */
.tabs::before,
.tabs::after,
.tab-wrapper::before,
.tab-wrapper::after,
.tab-nav-wrapper::before,
.tab-nav-wrapper::after,
.tab-container::before,
.tab-container::after,
.tab-nav::before,
.tab-nav::after,
div[role="tablist"]::before,
div[role="tablist"]::after,
.tabitem::before,
.tabitem::after,
button[role="tab"]::before,
button[role="tab"]::after,
button[role="tab"].selected::before,
button[role="tab"].selected::after {
    display: none !important;
    content: none !important;
    border: none !important;
    box-shadow: none !important;
    height: 0 !important;
    width: 0 !important;
}

/* Single Floating Segmented Pill Container */
.tab-nav,
div[role="tablist"],
div.tab-container[role="tablist"] {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
    background: var(--tab-nav-bg) !important;
    background-color: var(--tab-nav-bg) !important;
    border: 1px solid var(--tab-nav-border) !important;
    border-radius: 9999px !important;
    padding: 5px !important;
    margin: 0 auto !important;
    width: fit-content !important;
    max-width: fit-content !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    box-shadow: var(--tab-nav-shadow) !important;
    position: relative !important;
    z-index: 10 !important;
    transition: background 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease !important;
}

.tab-nav::-webkit-scrollbar,
div[role="tablist"]::-webkit-scrollbar {
    display: none !important;
}

/* Tab Item Buttons */
.tab-nav button,
button[role="tab"],
div.tab-container[role="tablist"] button[role="tab"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
    font-size: 0.90rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    color: var(--tab-btn-color) !important;
    border-radius: 9999px !important;
    padding: 9px 22px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    white-space: nowrap !important;
    user-select: none !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Hide unwanted Gradio overflow dropdown button */
button[aria-label="More"],
button[aria-label="More tabs"],
.tab-nav-more,
.tab-more-btn,
div[role="tablist"] > button:not([role="tab"]),
.tab-nav > button:not([role="tab"]) {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Inactive Tab Hover */
.tab-nav button:hover:not(.selected):not([aria-selected="true"]),
button[role="tab"]:hover:not(.selected):not([aria-selected="true"]),
div.tab-container[role="tablist"] button[role="tab"]:hover:not(.selected):not([aria-selected="true"]) {
    color: var(--tab-btn-hover-color) !important;
    background: var(--tab-btn-hover-bg) !important;
    border-color: var(--border-subtle) !important;
    transform: translateY(-1px) !important;
}

/* Active / Selected Tab Pill */
.tab-nav button.selected,
.tab-nav button[aria-selected="true"],
button[role="tab"].selected,
button[role="tab"][aria-selected="true"],
div.tab-container[role="tablist"] button[role="tab"].selected,
div.tab-container[role="tablist"] button[role="tab"][aria-selected="true"] {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    background: var(--tab-btn-active-bg) !important;
    border: 1px solid var(--tab-btn-active-border) !important;
    border-radius: 9999px !important;
    box-shadow: var(--tab-btn-active-shadow) !important;
    font-weight: 700 !important;
}

/* Cards & Inputs */
.gr-box, .gr-panel, .panel, .block:not(.hide-container):not(.auto-margin):not(:has(.prose)):not(:has(.md)) {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-card) !important;
}

/* ─── Markdown Blocks Clean Reset (No Box/Border Around Headers & Text) ───── */
.block.hide-container,
div.hide-container,
.block.hide-container.padded,
.block.auto-margin,
.block:has(.prose),
.block:has(.md),
.block:has(h1),
.block:has(h2),
.block:has(h3),
.block:has(h4),
.block:has(h5),
.block:has(h6),
div[data-testid="markdown-wrapper"],
div[class*="markdown"],
.prose,
.md {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    border-width: 0 !important;
    box-shadow: none !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}

/* Crisp & Modern Typography */
h1, h2, h3, h4, h5, h6 {
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

h1:not(#header-container h1) {
    background: transparent !important;
}

h3 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    margin: 14px 0 8px 0 !important;
    letter-spacing: -0.01em !important;
}

h4 {
    font-size: 0.96rem !important;
    font-weight: 600 !important;
    color: #cbd5e1 !important;
    margin: 8px 0 6px 0 !important;
    letter-spacing: 0 !important;
}

p {
    color: #94a3b8 !important;
    font-size: 0.92rem !important;
    line-height: 1.55 !important;
    margin: 4px 0 8px 0 !important;
}

hr {
    border: none !important;
    border-top: 1px solid rgba(255, 255, 255, 0.08) !important;
    margin: 18px 0 !important;
    background: transparent !important;
}

input:not([type="checkbox"]):not([type="radio"]), textarea, select, .gr-input, .gr-textbox textarea {
    background: var(--bg-input) !important;
    border: 1px solid rgba(255, 255, 255, 0.11) !important;
    border-radius: var(--radius-input) !important;
    color: var(--text-primary) !important;
    font-size: 0.95rem !important;
    font-family: inherit !important;
}

input:not([type="checkbox"]):not([type="radio"]):focus, textarea:focus, select:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.18) !important;
    outline: none !important;
}

/* ─── Checkbox & Toggles ─────────────────────────────────────────────────── */
.checkbox-container, label.checkbox-container, .gr-checkbox {
    display: inline-flex !important;
    align-items: center !important;
    gap: 12px !important;
    cursor: pointer !important;
    padding: 12px 18px !important;
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 14px !important;
    transition: all 0.2s ease !important;
    user-select: none !important;
}

.checkbox-container:hover, label.checkbox-container:hover {
    background: var(--card-bg) !important;
    border-color: var(--border-focus) !important;
    box-shadow: var(--card-shadow) !important;
}

input[type="checkbox"], .checkbox-container input[type="checkbox"], input[data-testid="checkbox"] {
    -webkit-appearance: none !important;
    appearance: none !important;
    width: 22px !important;
    height: 22px !important;
    min-width: 22px !important;
    min-height: 22px !important;
    border: 2px solid var(--border-focus) !important;
    border-radius: 6px !important;
    background: var(--input-bg) !important;
    background-color: var(--input-bg) !important;
    cursor: pointer !important;
    display: inline-grid !important;
    place-content: center !important;
    margin: 0 !important;
    padding: 0 !important;
    transition: all 0.15s ease !important;
    position: relative !important;
}

input[type="checkbox"]:hover {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 10px rgba(2, 132, 199, 0.3) !important;
}

input[type="checkbox"]:checked, input[data-testid="checkbox"]:checked {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
    background-color: #0284c7 !important;
    border-color: #0284c7 !important;
    box-shadow: 0 0 10px rgba(2, 132, 199, 0.4) !important;
}

input[type="checkbox"]:checked::before, input[data-testid="checkbox"]:checked::before {
    content: "✓" !important;
    color: #ffffff !important;
    font-size: 14px !important;
    font-weight: 900 !important;
    line-height: 1 !important;
    display: block !important;
    text-align: center !important;
}

.label-text, .checkbox-container span.label-text, .checkbox-container span {
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    font-size: 0.94rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
}

/* ─── Global Form Wrappers Reset ─────────────────────────────────────────── */
.form,
.form.svelte-d5xbca,
.tab1-card .form,
#model-select-row .form,
#api-input-row .form,
.gradio-row .form,
.gradio-column .form {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: visible !important;
    overflow-y: visible !important;
    overflow-x: visible !important;
}

/* ─── Dropdowns & Popups ─────────────────────────────────────────────────── */
/* ─── Dropdown Controls & Form Inputs ────────────────────────────────────── */
.gradio-dropdown,
.gradio-dropdown.block,
.gradio-dropdown .wrap,
.gradio-dropdown .wrap-inner,
div[data-testid="dropdown"],
div[data-testid="dropdown"] .wrap,
.wrap.svelte-1xfsv4t,
#model-dropdown,
#model-dropdown.block,
#model-dropdown .wrap,
#model-dropdown .wrap.default {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: visible !important;
    overflow-y: visible !important;
    overflow-x: visible !important;
    position: relative !important;
}

.secondary-wrap.svelte-1xfsv4t,
.gradio-dropdown .secondary-wrap,
#model-dropdown .secondary-wrap {
    display: flex !important;
    align-items: center !important;
    background: var(--input-bg) !important;
    background-color: var(--input-bg) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: var(--radius-input) !important;
    padding: 0 12px !important;
    height: 48px !important;
    min-height: 48px !important;
    box-shadow: var(--input-shadow) !important;
    position: relative !important;
    transition: all 0.2s ease !important;
}

.secondary-wrap.svelte-1xfsv4t:hover,
.secondary-wrap.svelte-1xfsv4t:focus-within,
#model-dropdown .secondary-wrap:hover,
#model-dropdown .secondary-wrap:focus-within,
#model-dropdown:focus-within .secondary-wrap {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.22), var(--input-shadow) !important;
}

#model-dropdown input,
.gradio-dropdown input,
input[role="combobox"],
.secondary-wrap input,
.secondary-wrap input.border-none,
.secondary-wrap input:focus,
.secondary-wrap input.border-none:focus {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    font-size: 0.94rem !important;
    font-weight: 500 !important;
    padding: 8px 40px 8px 8px !important; /* 40px right padding prevents any text/arrow collision */
    width: 100% !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    overflow: hidden !important;
}

.icon-wrap.svelte-1xfsv4t,
.secondary-wrap .icon-wrap {
    position: absolute !important;
    right: 14px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    pointer-events: all !important;
    cursor: pointer !important;
    color: var(--accent-blue) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.dropdown-arrow, .icon-wrap svg {
    fill: #38bdf8 !important;
    color: #38bdf8 !important;
    transition: transform 0.2s ease !important;
}

/* ─── Dropdown Popup Menu (Options List) ─────────────────────────────────── */
ul.options,
ul.options.svelte-1ou0lab,
[role="listbox"].options,
div.options,
.options-wrap {
    position: absolute !important;
    top: calc(100% + 6px) !important;
    bottom: auto !important;
    left: 0 !important;
    right: 0 !important;
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
    max-height: 280px !important;
    overflow-y: auto !important;
    background: var(--dropdown-menu-bg) !important;
    background-color: var(--dropdown-menu-bg) !important;
    opacity: 1 !important;
    border: 1px solid var(--dropdown-border) !important;
    border-radius: 14px !important;
    box-shadow: var(--dropdown-shadow) !important;
    padding: 6px !important;
    z-index: 999999 !important;
    margin-top: 0 !important;
}

ul.options li,
li[role="option"],
li.item.svelte-1ou0lab,
.options li {
    background: transparent !important;
    background-color: transparent !important;
    color: var(--dropdown-item-color) !important;
    -webkit-text-fill-color: var(--dropdown-item-color) !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    margin: 3px 0 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    display: flex !important;
    align-items: center !important;
}

ul.options li:hover,
li[role="option"]:hover,
li.item:hover {
    background: var(--dropdown-item-hover-bg) !important;
    background-color: var(--dropdown-item-hover-bg) !important;
    color: var(--dropdown-item-hover) !important;
    -webkit-text-fill-color: var(--dropdown-item-hover) !important;
}

ul.options li.selected,
li[role="option"].selected,
li.item.selected,
li[aria-selected="true"] {
    background: var(--dropdown-item-sel-bg) !important;
    background-color: var(--dropdown-item-sel-bg) !important;
    color: var(--dropdown-item-hover) !important;
    -webkit-text-fill-color: var(--dropdown-item-hover) !important;
    font-weight: 600 !important;
}

/* ─── Global Inline Code Tags ────────────────────────────────────────────── */
code, .md code, .prose code, span code, p code,
.gradio-markdown code, .markdown code, .bot-row .bot code {
    background: var(--code-bg) !important;
    background-color: var(--code-bg) !important;
    color: var(--code-color) !important;
    -webkit-text-fill-color: var(--code-color) !important;
    padding: 2px 8px !important;
    border-radius: 6px !important;
    border: 1px solid var(--code-border) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88em !important;
    font-weight: 600 !important;
}

/* ─── Unified Universal Button System ────────────────────────────────────── */
button, .gr-button {
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-sizing: border-box !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    cursor: pointer !important;
    letter-spacing: -0.01em !important;
}

/* Primary Green / Emerald Action: Ingest */
#btn-ingest, button#btn-ingest {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    background-color: #059669 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 0.98rem !important;
    height: 48px !important;
    min-height: 48px !important;
    padding: 0 24px !important;
    border: none !important;
    box-shadow: 0 4px 18px rgba(16, 185, 129, 0.40) !important;
}

#btn-ingest:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 26px rgba(16, 185, 129, 0.55) !important;
}

/* Primary Electric Sky-Blue Action: Send */
#btn-send, button#btn-send {
    background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%) !important;
    background-color: #0284c7 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 0.96rem !important;
    height: 48px !important;
    min-height: 48px !important;
    padding: 0 24px !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(56, 189, 248, 0.35) !important;
}

#btn-send:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(56, 189, 248, 0.52) !important;
}

/* Primary Mint-Cyan Action: Connect */
#btn-connect, button#btn-connect {
    background: linear-gradient(135deg, #0d9488 0%, #10b981 100%) !important;
    background-color: #0d9488 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 0.94rem !important;
    font-weight: 700 !important;
    height: 48px !important;
    min-height: 48px !important;
    margin: 0 !important;
    padding: 0 22px !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 16px rgba(16, 185, 129, 0.35) !important;
    white-space: nowrap !important;
}

#btn-connect:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.5) !important;
}

/* ─── Tab 1 Symmetrical Glassmorphic Dual Cards ────────────────────────── */
#tab1-main-row {
    position: relative !important;
    z-index: 1000 !important;
}

.tab1-card {
    background: var(--card-bg) !important;
    background-color: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    box-shadow: var(--card-shadow) !important;
    backdrop-filter: blur(16px) !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !imp/* ─── File Upload Outer Component ────────────────────────────────────────── */
#file-upload,
div[data-testid="file-upload"],
div[data-testid="file-upload"]#file-upload {
    background: var(--card-bg) !important;
    background-color: var(--card-bg) !important;
    border: 1.5px dashed var(--accent-blue) !important;
    border-radius: 14px !important;
    color: var(--text-primary) !important;
    overflow: hidden !important;
    position: relative !important;
    min-height: 185px !important;
    height: auto !important;
    box-sizing: border-box !important;
    cursor: pointer !important;
    box-shadow: var(--card-shadow) !important;
    transition: all 0.2s ease !important;
}

#file-upload:hover {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.20), var(--input-shadow) !important;
}

/* Floating label on top of file upload */
#file-upload label,
label.svelte-19djge9,
label.float,
label[data-testid="block-label"],
.block-label {
    background: var(--card-bg) !important;
    background-color: var(--card-bg) !important;
    color: var(--accent-blue) !important;
    -webkit-text-fill-color: var(--accent-blue) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    padding: 3px 12px !important;
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    pointer-events: none !important;
    user-select: none !important;
    position: absolute !important;
    top: 8px !important;
    left: 10px !important;
    z-index: 15 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}

/* Main empty dropzone button (only when no file is uploaded) */
#file-upload button.center.boundedheight:not(.icon-mode):not(.icon-button),
#file-upload .upload-container {
    width: 100% !important;
    min-height: 185px !important;
    height: 100% !important;
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    padding: 24px 12px !important;
    color: var(--text-primary) !important;
}

#file-upload .upload-container svg {
    stroke: var(--accent-blue) !important;
    fill: var(--accent-blue) !important;
}

#file-upload .upload-container span {
    color: var(--text-primary) !important;
}

/* Top-right corner controls when files are uploaded (upload more & clear) */
#file-upload .icon-button-wrapper,
#file-upload .top-panel {
    position: absolute !important;
    top: 8px !important;
    right: 8px !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    height: auto !important;
    min-height: auto !important;
    width: auto !important;
    z-index: 20 !important;
}

#file-upload button.icon-button,
#file-upload .icon-button {
    position: relative !important;
    background: var(--card-bg) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    width: 30px !important;
    height: 30px !important;
    min-height: 30px !important;
    max-height: 30px !important;
    min-width: 30px !important;
    max-width: 30px !important;
    padding: 0 !important;
    margin: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: var(--text-secondary) !important;
    box-shadow: var(--card-shadow) !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}

#file-upload button.icon-button::before,
#file-upload button.icon-button::after,
#file-upload .icon-button-wrapper::before,
#file-upload .icon-button-wrapper::after {
    display: none !important;
    content: none !important;
    border: none !important;
    background: transparent !important;
    width: 0 !important;
    height: 0 !important;
}

#file-upload button.icon-button:hover,
#file-upload button.icon-button[aria-label="common.upload"]:hover,
#file-upload button.icon-button[title="common.upload"]:hover {
    border-color: var(--accent-blue) !important;
    color: var(--accent-blue) !important;
    background: var(--dropdown-item-hover-bg) !important;
    transform: translateY(-1px) !important;
}

#file-upload button.icon-button[aria-label="Clear"]:hover,
#file-upload button.icon-button[title="Clear"]:hover {
    border-color: rgba(239, 68, 68, 0.5) !important;
    color: #f87171 !important;
    background: rgba(239, 68, 68, 0.15) !important;
    transform: translateY(-1px) !important;
}

#file-upload button.icon-mode {
    display: block !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 100% !important;
    max-height: 100% !important;
    opacity: 0 !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    cursor: pointer !important;
    z-index: 10 !important;
    background: transparent !important;
    pointer-events: auto !important;
}

#file-upload button.icon-button svg,
#file-upload button.icon-button .small {
    pointer-events: none !important;
}

/* File list table & container */
#file-upload .file-preview-holder {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    min-height: 130px !important;
    max-height: 185px !important;
    height: auto !important;
    overflow-y: auto !important;
    padding: 42px 10px 10px 10px !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

#file-upload .file-preview-holder::-webkit-scrollbar {
    width: 5px !important;
}

#file-upload .file-preview-holder::-webkit-scrollbar-track {
    background: transparent !important;
}

#file-upload .file-preview-holder::-webkit-scrollbar-thumb {
    background: var(--border-focus) !important;
    border-radius: 4px !important;
}

#file-upload table.file-preview,
.file-preview table,
.file-preview-holder table {
    width: 100% !important;
    border-collapse: separate !important;
    border-spacing: 0 5px !important;
    background: transparent !important;
    border: none !important;
}

#file-upload tr.file,
#file-upload tr,
.file-preview tr,
.file-preview-holder tr,
.file-preview-holder .file,
.file-preview-holder .file-item {
    background: var(--card-bg) !important;
    background-color: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 9px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    transition: all 0.15s ease !important;
}

#file-upload tr.file:hover,
#file-upload tr:hover,
.file-preview tr:hover,
.file-preview-holder tr:hover,
.file-preview-holder .file:hover {
    background: var(--dropdown-item-hover-bg) !important;
    border-color: var(--border-focus) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06) !important;
}

#file-upload td.filename,
.file-preview td.filename,
.file-preview-holder td.filename,
.file-preview-holder .filename {
    padding: 8px 12px !important;
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border: none !important;
}

#file-upload td.download,
.file-preview td.download,
.file-preview-holder td.download {
    padding: 8px 12px !important;
    text-align: right !important;
    border: none !important;
}

#file-upload a.download-link,
#file-upload .download-link,
.file-preview-holder .download-link {
    color: var(--accent-green) !important;
    -webkit-text-fill-color: var(--accent-green) !important;
    font-weight: 600 !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    text-decoration: none !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
}

#file-upload a.download-link:hover {
    color: var(--accent-blue) !important;
    -webkit-text-fill-color: var(--accent-blue) !important;
    text-decoration: underline !important;
}

#file-upload button[aria-label="Remove file"],
#file-upload button[aria-label="Delete"],
#file-upload button[aria-label="Clear"],
#file-upload .clear-button,
.file-preview-holder button {
    color: var(--text-secondary) !important;
    -webkit-text-fill-color: var(--text-secondary) !important;
    cursor: pointer !important;
    transition: color 0.15s ease !important;
}

#file-upload button[aria-label="Remove file"]:hover,
#file-upload button[aria-label="Delete"]:hover,
#file-upload button[aria-label="Clear"]:hover,
.file-preview-holder button:hover {
    color: #ef4444 !important;
    -webkit-text-fill-color: #ef4444 !important;
}
/* ─── Tab 2 Input Cards & Action Row Perfect Alignment ───────────────────── */
#tab2-inputs-row {
    display: flex !important;
    gap: 16px !important;
    margin-bottom: 8px !important;
    align-items: stretch !important;
}

#file-upload,
div[data-testid="file-upload"]#file-upload {
    height: auto !important;
    min-height: 185px !important;
    max-height: 240px !important;
    box-sizing: border-box !important;
    margin-bottom: 0 !important;
}

#url-input-box,
div[data-testid="textbox"]#url-input-box {
    height: auto !important;
    min-height: 185px !important;
    max-height: 240px !important;
    box-sizing: border-box !important;
    margin-bottom: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    background: var(--card-bg) !important;
    background-color: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 14px !important;
    padding: 12px 14px !important;
    box-shadow: var(--card-shadow) !important;
}

#url-input-box label {
    margin-bottom: 6px !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

#url-input-box textarea {
    height: 110px !important;
    min-height: 110px !important;
    resize: none !important;
    box-sizing: border-box !important;
    background: var(--input-bg) !important;
    background-color: var(--input-bg) !important;
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    line-height: 1.4 !important;
    box-shadow: var(--input-shadow) !important;
}

#ingest-action-row {
    display: flex !important;
    align-items: center !important;
    gap: 16px !important;
    margin-top: 12px !important;
    margin-bottom: 12px !important;
}

#append-toggle,
#append-toggle.block,
div[data-testid="checkbox"]#append-toggle {
    height: 48px !important;
    min-height: 48px !important;
    max-height: 48px !important;
    display: flex !important;
    align-items: center !important;
    padding: 0 16px !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    background: var(--card-bg) !important;
    background-color: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 12px !important;
    box-shadow: var(--card-shadow) !important;
    transition: all 0.2s ease !important;
}

#append-toggle:hover {
    border-color: var(--accent-blue) !important;
}

#append-toggle label,
#append-toggle label.checkbox-container,
#append-toggle .checkbox-container {
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
    gap: 12px !important;
    width: 100% !important;
    height: 100% !important;
    cursor: pointer !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text-primary) !important;
}

.nav-hint-banner {
    background: rgba(16, 185, 129, 0.12) !important;
    border: 1px solid rgba(16, 185, 129, 0.35) !important;
    border-radius: 12px !important;
    padding: 12px 18px !important;
    color: #047857 !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    margin: 12px 0 !important;
    box-shadow: 0 2px 10px rgba(16, 185, 129, 0.10) !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}

body.dark .nav-hint-banner, .dark .nav-hint-banner {
    color: #34d399 !important;
    background: rgba(16, 185, 129, 0.16) !important;
    border-color: rgba(52, 211, 153, 0.40) !important;
}ax-height: 160px !important;
    box-sizing: border-box !important;
    margin-bottom: 0 !important;
}

#url-input-box,
div[data-testid="textbox"]#url-input-box {
    height: 160px !important;
    min-height: 160px !important;
    max-height: 160px !important;
    box-sizing: border-box !important;
    margin-bottom: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    background: var(--card-bg) !important;
    background-color: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 14px !important;
    padding: 12px 14px !important;
    box-shadow: var(--card-shadow) !important;
}

#url-input-box label {
    margin-bottom: 6px !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

#url-input-box textarea {
    height: 96px !important;
    min-height: 96px !important;
    max-height: 96px !important;
    resize: none !important;
    box-sizing: border-box !important;
    background: var(--input-bg) !important;
    background-color: var(--input-bg) !important;
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    line-height: 1.4 !important;
    box-shadow: var(--input-shadow) !important;
}

#ingest-action-row {
    display: flex !important;
    align-items: center !important;
    gap: 16px !important;
    margin-top: 12px !important;
    margin-bottom: 12px !important;
}

#append-toggle,
#append-toggle.block,
div[data-testid="checkbox"]#append-toggle {
    height: 48px !important;
    min-height: 48px !important;
    max-height: 48px !important;
    display: flex !important;
    align-items: center !important;
    padding: 0 16px !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    background: var(--card-bg) !important;
    background-color: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 12px !important;
    box-shadow: var(--card-shadow) !important;
    transition: all 0.2s ease !important;
}

#append-toggle:hover {
    border-color: var(--accent-blue) !important;
}

#append-toggle label,
#append-toggle label.checkbox-container,
#append-toggle .checkbox-container {
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
    gap: 12px !important;
    width: 100% !important;
    height: 100% !important;
    cursor: pointer !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
}

#append-toggle span {
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    font-weight: 500 !important;
}

#append-toggle label:hover,
#append-toggle label.checkbox-container:hover,
#append-toggle .checkbox-container:hover {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

#ingest-btn-subrow {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    height: 48px !important;
    min-height: 48px !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
}

#btn-ingest {
    flex: 1.4 1 0 !important;
    height: 48px !important;
    min-height: 48px !important;
    max-height: 48px !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    white-space: nowrap !important;
    font-size: 0.90rem !important;
    font-weight: 600 !important;
    padding: 0 16px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

#btn-reset-kb {
    flex: 1 1 0 !important;
    height: 48px !important;
    min-height: 48px !important;
    max-height: 48px !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    white-space: nowrap !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 0 14px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Primary Indigo-Violet Benchmark Actions: Hallucination & Top-k */
#btn-halluc, button#btn-halluc,
#btn-topk, button#btn-topk {
    background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%) !important;
    background-color: #6366f1 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 0.96rem !important;
    height: 48px !important;
    min-height: 48px !important;
    width: 100% !important;
    padding: 0 20px !important;
    border: none !important;
    box-shadow: 0 4px 18px rgba(99, 102, 241, 0.35) !important;
}

#btn-halluc:hover, #btn-topk:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 26px rgba(99, 102, 241, 0.52) !important;
}

/* Universal Secondary Dark Buttons: Reset KB & Secondary actions */
#btn-reset-kb, button#btn-reset-kb,
button.secondary, .gr-button-secondary {
    background: rgba(30, 41, 59, 0.75) !important;
    background-color: rgba(30, 41, 59, 0.75) !important;
    color: #cbd5e1 !important;
    -webkit-text-fill-color: #cbd5e1 !important;
    font-size: 0.94rem !important;
    font-weight: 600 !important;
    height: 48px !important;
    min-height: 48px !important;
    padding: 0 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
}

#btn-reset-kb:hover, button#btn-reset-kb:hover,
button.secondary:hover {
    background: rgba(239, 68, 68, 0.18) !important;
    background-color: rgba(239, 68, 68, 0.18) !important;
    border-color: rgba(239, 68, 68, 0.45) !important;
    color: #f87171 !important;
    -webkit-text-fill-color: #f87171 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(239, 68, 68, 0.25) !important;
}

/* ─── Chat Input Row Alignment & Styling ─────────────────────────────────── */
#chat-input-row {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    margin-top: 14px !important;
    width: 100% !important;
}

#chat-input-row > div:first-child {
    flex: 1 1 auto !important;
}

#chat-input-row .form,
#chat-input-row div.form {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Eliminate Gradio's outer block card wrapper around msg-input (prevents box-in-box double borders) */
#msg-input,
#msg-input.block,
div[data-testid="textbox"]#msg-input,
#chat-input-row #msg-input .wrap {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

#chat-input-row #msg-input label.container {
    background: var(--input-bg) !important;
    background-color: var(--input-bg) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 14px !important;
    box-shadow: var(--input-shadow) !important;
    padding: 0 !important;
    margin: 0 !important;
    transition: all 0.2s ease !important;
    display: flex !important;
    align-items: center !important;
    min-height: 48px !important;
    height: 48px !important;
    box-sizing: border-box !important;
}

#chat-input-row #msg-input label.container:focus-within {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2), var(--input-shadow) !important;
}

#chat-input-row #msg-input .input-container {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    width: 100% !important;
}

#chat-input-row #msg-input textarea {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    padding: 12px 18px !important;
    min-height: 46px !important;
    height: 46px !important;
    box-sizing: border-box !important;
    resize: none !important;
    width: 100% !important;
}

#btn-send, button#btn-send {
    background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%) !important;
    background-color: #0284c7 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    height: 48px !important;
    min-height: 48px !important;
    padding: 0 22px !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 16px rgba(56, 189, 248, 0.35) !important;
    flex: 0 0 auto !important;
    white-space: nowrap !important;
}

#btn-clear, button#btn-clear {
    background: var(--card-bg) !important;
    background-color: var(--card-bg) !important;
    color: var(--text-secondary) !important;
    -webkit-text-fill-color: var(--text-secondary) !important;
    font-size: 1.25rem !important;
    height: 48px !important;
    min-height: 48px !important;
    width: 48px !important;
    min-width: 48px !important;
    max-width: 48px !important;
    padding: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 12px !important;
    box-shadow: var(--card-shadow) !important;
    flex: 0 0 48px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

#btn-clear:hover, button#btn-clear:hover {
    background: var(--input-bg) !important;
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
}

/* ─── Hide default corner Clear/Trash button on Chatbot ──────────────────── */
#chatbot-box .icon-button-wrapper,
#chatbot-box .top-panel,
#chatbot-box button[aria-label="Clear"],
#chatbot-box button[title="Clear"],
.gradio-chatbot .icon-button-wrapper,
.gradio-chatbot .top-panel,
.gradio-chatbot button[aria-label="Clear"],
.gradio-chatbot button[title="Clear"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
}


/* Evaluation Action Column & Card */
.eval-action-col {
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    height: 100% !important;
}

.eval-info-card {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 16px 18px !important;
    margin-bottom: 12px !important;
    box-sizing: border-box !important;
}

.eval-info-header {
    display: flex !important;
    align-items: center !important;
    margin-bottom: 8px !important;
}

.eval-info-title {
    font-size: 1.02rem !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    letter-spacing: -0.01em !important;
}

.eval-info-text {
    font-size: 0.88rem !important;
    color: #94a3b8 !important;
    line-height: 1.55 !important;
    margin: 0 !important;
}

/* Chatbot & Bubbles */
#chatbot-box,
.gradio-chatbot {
    border-radius: 20px !important;
    border: 1px solid var(--card-border) !important;
    background: var(--card-bg) !important;
    background-color: var(--card-bg) !important;
    box-shadow: var(--card-shadow) !important;
}

/* User Message: Single-Box Blue Gradient */
.user-row .flex-wrap > .user {
    background: var(--tab-btn-active-bg) !important;
    border: none !important;
    border-radius: 20px 20px 4px 20px !important;
    box-shadow: 0 4px 18px rgba(0, 122, 255, 0.35) !important;
    padding: 0 !important;
    overflow: hidden !important;
}

.user-row .user > div,
.user-row .user .message,
.user-row .user .panel-full-width,
.user-row .user [class*="panel"],
.user-row .user .message-content {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 12px 18px !important;
    margin: 0 !important;
}

.user-row .user * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 0.96rem !important;
    font-weight: 500 !important;
}

/* Assistant Message: Clean Glass Card */
.bot-row .flex-wrap > .bot {
    background: var(--bot-bubble-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 20px 20px 20px 4px !important;
    box-shadow: var(--card-shadow) !important;
    backdrop-filter: blur(20px) !important;
    padding: 0 !important;
    overflow: hidden !important;
}

.bot-row .bot > div,
.bot-row .bot .message,
.bot-row .bot .panel-full-width,
.bot-row .bot [class*="panel"],
.bot-row .bot .message-content {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 16px 20px !important;
    margin: 0 !important;
}

.bot-row .bot p, .bot-row .bot span, .bot-row .bot li {
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    font-size: 0.94rem !important;
    line-height: 1.7 !important;
}

.bot-row .bot h1, .bot-row .bot h2, .bot-row .bot h3 {
    color: var(--accent-blue) !important;
    -webkit-text-fill-color: var(--accent-blue) !important;
    font-weight: 700 !important;
}

.bot-row .bot code {
    background: rgba(56, 189, 248, 0.12) !important;
    color: var(--accent-blue) !important;
    padding: 2px 6px !important;
    border-radius: 5px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Table Cards */
.custom-table-card {
    background: var(--card-bg) !important;
    background-color: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    margin: 12px 0 !important;
    box-shadow: var(--card-shadow) !important;
}

.custom-eval-table {
    width: 100% !important;
    border-collapse: collapse !important;
    font-size: 0.88rem !important;
}

.custom-eval-table th {
    background: var(--input-bg) !important;
    color: var(--text-secondary) !important;
    -webkit-text-fill-color: var(--text-secondary) !important;
    font-weight: 700 !important;
    padding: 13px 18px !important;
    border-bottom: 1px solid var(--card-border) !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
}

.custom-eval-table th:first-child,
.custom-eval-table td:first-child {
    border-right: 1px solid var(--card-border) !important;
}

.custom-eval-table td {
    padding: 11px 18px !important;
    border-bottom: 1px solid var(--card-border) !important;
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    font-weight: 500 !important;
    transition: background 0.15s ease !important;
}

.custom-eval-table tbody tr:last-child td {
    border-bottom: none !important;
}

.custom-eval-table tr:hover td {
    background: var(--dropdown-item-hover-bg) !important;
}

/* Badges */
.badge {
    display: inline-flex !important;
    align-items: center !important;
    padding: 3px 10px !important;
    border-radius: 20px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}

.badge-green { background: rgba(16, 185, 129, 0.15) !important; color: #34d399 !important; border: 1px solid rgba(52, 211, 153, 0.3) !important; }
.badge-cyan  { background: rgba(56, 189, 248, 0.15) !important; color: #38bdf8 !important; border: 1px solid rgba(56, 189, 248, 0.3) !important; }
.badge-blue  { background: rgba(59, 130, 246, 0.15) !important; color: #60a5fa !important; border: 1px solid rgba(96, 165, 250, 0.3) !important; }
.badge-purple{ background: rgba(139, 92, 246, 0.15) !important; color: #c084fc !important; border: 1px solid rgba(192, 132, 252, 0.3) !important; }
.badge-slate { background: rgba(100, 116, 139, 0.15) !important; color: #94a3b8 !important; border: 1px solid rgba(148, 163, 184, 0.25) !important; }
.badge-red   { background: rgba(239, 68, 68, 0.15) !important; color: #f87171 !important; border: 1px solid rgba(248, 113, 113, 0.3) !important; }

.eval-placeholder {
    padding: 32px 20px !important;
    text-align: center !important;
    background: var(--input-bg) !important;
    background-color: var(--input-bg) !important;
    border: 1px dashed var(--border-subtle) !important;
    border-radius: 16px !important;
    margin: 12px 0 !important;
}

.eval-placeholder .placeholder-icon { font-size: 2rem !important; margin-bottom: 6px !important; }
.eval-placeholder .placeholder-title { font-size: 0.95rem !important; font-weight: 700 !important; color: var(--text-primary) !important; -webkit-text-fill-color: var(--text-primary) !important; }
.eval-placeholder .placeholder-desc { font-size: 0.84rem !important; color: var(--text-secondary) !important; margin-top: 4px !important; }

/* ─── Global Fluid Container & Alignment ──────────────────────────────────── */
.gradio-container {
    max-width: 1360px !important;
    margin: 0 auto !important;
    padding: 24px 20px 48px !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

div.tab-container.visually-hidden {
    display: none !important;
    height: 0 !important;
    overflow: hidden !important;
    position: absolute !important;
    pointer-events: none !important;
    visibility: hidden !important;
}

/* Hide tab overflow button on screens where all tabs fit */
@media (min-width: 769px) {
    .overflow-menu, button[aria-label="More tabs"], span.overflow-menu {
        display: none !important;
    }
}

.overflow-menu button, button[aria-label="More tabs"] {
    background: rgba(15, 23, 42, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #f1f5f9 !important;
    border-radius: 10px !important;
    padding: 6px 12px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.overflow-menu button:hover {
    background: rgba(56, 189, 248, 0.15) !important;
    border-color: var(--accent-blue) !important;
}

.overflow-dropdown, div.overflow-dropdown {
    background: #0f172a !important;
    background-color: #0f172a !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    border-radius: 14px !important;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.85) !important;
    padding: 6px !important;
    z-index: 999999 !important;
}

.overflow-dropdown button {
    display: block !important;
    width: 100% !important;
    text-align: left !important;
    padding: 10px 14px !important;
    font-size: 0.90rem !important;
    font-weight: 500 !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
    background: transparent !important;
    border: none !important;
    cursor: pointer !important;
    margin: 2px 0 !important;
    transition: all 0.15s ease !important;
}

.overflow-dropdown button:hover, .overflow-dropdown button.selected {
    background: rgba(56, 189, 248, 0.18) !important;
    color: #38bdf8 !important;
}

/* ─── Responsive Media Queries for All Devices ───────────────────────────── */

/* Laptops and iPad Pro Landscape (max-width: 1024px) */
@media (max-width: 1024px) {
    .gradio-container {
        padding: 18px 16px 36px !important;
    }
    #header-container {
        padding: 26px 20px 20px !important;
        margin-bottom: 20px !important;
    }
    #header-container h1 {
        font-size: 1.95rem !important;
    }
}

/* Tablets and Large Phones (max-width: 768px) */
@media (max-width: 768px) {
    .gradio-container {
        padding: 10px 8px 24px !important;
    }
    
    #header-container {
        padding: 52px 14px 18px !important;
        border-radius: 16px !important;
        margin-bottom: 16px !important;
        position: relative !important;
    }
    
    #theme-toggle-btn {
        top: 12px !important;
        right: 12px !important;
        padding: 5px 12px !important;
        font-size: 0.76rem !important;
    }
    
    #header-container h1 {
        font-size: 1.35rem !important;
        flex-direction: column !important;
        gap: 6px !important;
        margin: 0 0 8px !important;
    }
    
    #header-container .header-logo-icon {
        font-size: 2.2rem !important;
    }
    
    #header-container .header-title-text {
        font-size: 1.35rem !important;
        text-align: center !important;
    }
    
    #header-container p {
        font-size: 0.82rem !important;
        margin-bottom: 12px !important;
        padding: 0 4px !important;
    }
    
    .header-badges-row {
        gap: 6px !important;
        justify-content: center !important;
    }
    
    .badge-tag {
        font-size: 0.74rem !important;
        padding: 4px 10px !important;
    }
    
    .tab-wrapper,
    div.tabs > div:first-child,
    div[data-testid="tabs"] > div:first-child {
        margin: 0 auto 16px auto !important;
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
    }
    
    .tab-nav,
    div[role="tablist"],
    div.tab-container[role="tablist"] {
        border-radius: 9999px !important;
        max-width: 100% !important;
        width: auto !important;
        display: inline-flex !important;
        justify-content: center !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: none !important;
        padding: 4px !important;
        margin: 0 auto 16px auto !important;
    }
    
    .tab-nav button,
    .tab-container button,
    .tab-wrapper button,
    button[role="tab"],
    div.tab-container[role="tablist"] button[role="tab"] {
        padding: 7px 12px !important;
        font-size: 0.76rem !important;
        flex-shrink: 0 !important;
        border-radius: 9999px !important;
    }
    
    /* Stack form rows cleanly on mobile */
    .row, #tab1-main-row, #api-key-input-row {
        flex-direction: column !important;
        gap: 12px !important;
    }
    
    .column, div[class*="column"], .form, div.form {
        width: 100% !important;
        min-width: 100% !important;
    }

    #chat-input-row, div#chat-input-row {
        flex-direction: column !important;
        align-items: stretch !important;
        gap: 8px !important;
        width: 100% !important;
    }

    #chat-input-row > div:first-child,
    #chat-input-row #msg-input,
    #chat-input-row #msg-input label.container,
    #chat-input-row #msg-input .input-container {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Touch-friendly full-width action buttons on mobile */
    #btn-ingest, button#btn-ingest,
    #btn-send, button#btn-send,
    #btn-connect, button#btn-connect,
    #btn-halluc, button#btn-halluc,
    #btn-topk, button#btn-topk,
    #btn-clear, button#btn-clear,
    #btn-reset-kb, button#btn-reset-kb,
    .lg.secondary, button.lg.secondary {
        width: 100% !important;
        min-height: 46px !important;
        font-size: 0.92rem !important;
        justify-content: center !important;
    }
    
    .checkbox-container, label.checkbox-container {
        width: 100% !important;
        box-sizing: border-box !important;
        justify-content: flex-start !important;
    }
    
    #chatbot-box {
        min-height: 380px !important;
        height: 440px !important;
    }
    
    .user-row .user > div, .user-row .user .message-content,
    .bot-row .bot > div, .bot-row .bot .message-content {
        padding: 12px 14px !important;
    }
}

/* Compact Smartphones (max-width: 480px) */
@media (max-width: 480px) {
    .gradio-container {
        padding: 8px 4px 18px !important;
    }
    #header-container {
        padding: 50px 10px 14px !important;
    }
    #header-container h1 {
        font-size: 1.22rem !important;
    }
    #header-container .header-title-text {
        font-size: 1.22rem !important;
    }
    #header-container p {
        font-size: 0.76rem !important;
    }
    .tab-nav button, .tab-container button, .tab-wrapper button, button[role="tab"] {
        padding: 6px 10px !important;
        font-size: 0.74rem !important;
    }
}


"""

HEADER_HTML = """
<div id="header-container" style="position: relative;">
    <button id="theme-toggle-btn" onclick="(function(e){
        if(e) { e.preventDefault(); e.stopPropagation(); }
        var isDark = document.documentElement.classList.contains('dark') || 
                     document.body.classList.contains('dark') || 
                     (document.querySelector('.gradio-container') && document.querySelector('.gradio-container').classList.contains('dark')) ||
                     (document.documentElement.getAttribute('data-theme') === 'dark');
        var nextTheme = isDark ? 'light' : 'dark';
        var isNextLight = (nextTheme === 'light');
        
        var targets = [document.documentElement, document.body, document.querySelector('.gradio-container')].filter(Boolean);
        targets.forEach(function(el) {
            if (isNextLight) {
                el.classList.add('light-theme', 'light');
                el.classList.remove('dark', 'dark-theme');
                el.setAttribute('data-theme', 'light');
            } else {
                el.classList.remove('light-theme', 'light');
                el.classList.add('dark', 'dark-theme');
                el.setAttribute('data-theme', 'dark');
            }
        });
        try { localStorage.setItem('rag_theme', nextTheme); } catch(err){}
        
        var icon = document.getElementById('theme-icon');
        var text = document.getElementById('theme-text');
        if (icon) icon.textContent = isNextLight ? '🌙' : '☀️';
        if (text) text.textContent = isNextLight ? 'Dark Mode' : 'Light Mode';
    })(event)" title="Toggle Dark / Light Mode" aria-label="Toggle Theme">
        <span id="theme-icon">🌙</span>
        <span id="theme-text">Dark Mode</span>
    </button>
    <img src="data:image/svg+xml;utf8,<svg/>" onerror="(function(){
        try {
            var urlParams = new URLSearchParams(window.location.search);
            var urlTheme = urlParams.get('__theme');
            var saved = localStorage.getItem('rag_theme');
            var active = saved || urlTheme || 'light';
            var isDark = (active === 'dark');
            var targets = [document.documentElement, document.body, document.querySelector('.gradio-container')].filter(Boolean);
            targets.forEach(function(el) {
                if (isDark) {
                    el.classList.add('dark', 'dark-theme');
                    el.classList.remove('light', 'light-theme');
                    el.setAttribute('data-theme', 'dark');
                } else {
                    el.classList.add('light', 'light-theme');
                    el.classList.remove('dark', 'dark-theme');
                    el.setAttribute('data-theme', 'light');
                }
            });
            var icon = document.getElementById('theme-icon');
            var text = document.getElementById('theme-text');
            if (icon) icon.textContent = isDark ? '☀️' : '🌙';
            if (text) text.textContent = isDark ? 'Light Mode' : 'Dark Mode';
        } catch(e){}
    })()" style="display:none;" />
    <h1>
        <span class="header-logo-icon">🤖</span>
        <span class="header-title-text">RAG AI Knowledge Assistant</span>
    </h1>
    <p>2026 World-Class Multi-Modal Pipeline · Documents & Live Web URLs · Grounded Citations & Metrics</p>
    <div class="header-badges-row">
        <span class="badge-tag badge-fast">⚡ Ultra-Fast Streaming</span>
        <span class="badge-tag badge-search">🔍 Hybrid FAISS + BM25 Search</span>
        <span class="badge-tag badge-guard">🛡️ Zero-Hallucination Guard</span>
        <span class="badge-tag badge-ingest">🌐 Multi-URL & File Ingestion</span>
    </div>
</div>
"""

CUSTOM_JS = """
() => {
    // 1. Theme Management
    function applyThemeState(theme) {
        var isLight = (theme === 'light');
        var targets = [document.documentElement, document.body, document.querySelector('.gradio-container')].filter(Boolean);
        targets.forEach(function(el) {
            if (isLight) {
                el.classList.add('light-theme', 'light');
                el.classList.remove('dark', 'dark-theme');
                el.setAttribute('data-theme', 'light');
            } else {
                el.classList.remove('light-theme', 'light');
                el.classList.add('dark', 'dark-theme');
                el.setAttribute('data-theme', 'dark');
            }
        });
        
        var icon = document.getElementById('theme-icon');
        var text = document.getElementById('theme-text');
        if (icon) icon.textContent = isLight ? '🌙' : '☀️';
        if (text) text.textContent = isLight ? 'Dark Mode' : 'Light Mode';
    }

    function initTheme() {
        try {
            var urlParams = new URLSearchParams(window.location.search);
            var urlTheme = urlParams.get('__theme');
            var savedTheme = localStorage.getItem('rag_theme');
            var activeTheme = savedTheme || urlTheme || 'light';
            applyThemeState(activeTheme);
        } catch(e){}
    }
    
    initTheme();
    setTimeout(initTheme, 100);
    setTimeout(initTheme, 400);

    // 2. Global Delegated Click Listener
    document.addEventListener('click', function(e) {
        // Theme toggle button click
        var themeBtn = e.target.closest('#theme-toggle-btn');
        if (themeBtn) {
            e.preventDefault();
            e.stopPropagation();
            var isDark = document.documentElement.classList.contains('dark') || 
                         document.body.classList.contains('dark') || 
                         (document.querySelector('.gradio-container') && document.querySelector('.gradio-container').classList.contains('dark')) ||
                         (document.documentElement.getAttribute('data-theme') === 'dark');
            var nextTheme = isDark ? 'light' : 'dark';
            try { localStorage.setItem('rag_theme', nextTheme); } catch(err){}
            applyThemeState(nextTheme);
            return;
        }

        // File upload button click
        var uploadBtn = e.target.closest('#file-upload button[aria-label="common.upload"]') || e.target.closest('#file-upload button[title="common.upload"]');
        if (uploadBtn) {
            var fi = uploadBtn.querySelector('input[type="file"]') || document.querySelector('#file-upload input[type="file"]');
            if (fi && !window.__preventRecursion) {
                window.__preventRecursion = true;
                fi.click();
                setTimeout(function() { window.__preventRecursion = false; }, 200);
            }
            return;
        }

        var fu = e.target.closest('#file-upload') || e.target.closest('div[data-testid="file-upload"]');
        if (fu && e.target.tagName !== 'INPUT' && !e.target.closest('button[aria-label="Remove file"]') && !e.target.closest('button[aria-label="Clear"]') && !e.target.closest('button[aria-label="Delete"]') && !e.target.closest('a') && !e.target.closest('.file-preview-holder')) {
            var fi = fu.querySelector('input[type="file"]');
            if (fi && !window.__preventRecursion) {
                window.__preventRecursion = true;
                fi.click();
                setTimeout(function() { window.__preventRecursion = false; }, 200);
            }
        }
    }, true);
}
"""

with gr.Blocks(
    title="RAG AI Knowledge Assistant — 2026 Edition",
    theme=gr.themes.Base(
        primary_hue="sky",
        secondary_hue="emerald",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Plus Jakarta Sans")],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono")],
    ),
    css=CSS,
    js=CUSTOM_JS,
) as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ── TAB 1: SETUP & MODEL ──────────────────────────────────────────────
        with gr.TabItem("⚙️ Setup & Model Selector"):
            env_key = os.environ.get("NVIDIA_API_KEY", "").strip()
            if env_key:
                masked_env = env_key[:10] + "..." + env_key[-4:] if len(env_key) > 14 else "Active"
                initial_api_status = format_api_status_html(True, masked_key=masked_env, model_id=CONFIG["llm_model"])
                initial_key_val = env_key
            else:
                initial_api_status = format_api_status_html(False, "Please enter your NVIDIA API key below and click Connect.")
                initial_key_val = ""

            with gr.Row(equal_height=True, elem_id="tab1-main-row"):
                with gr.Column(scale=1, elem_classes=["tab1-card"]):
                    gr.HTML("""
                    <div class="tab1-card-header">
                        <div class="tab1-card-title">🔑 NVIDIA NIM API Credentials</div>
                        <div class="tab1-card-desc">Get your free API key at <a href="https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5" target="_blank" style="color: #38bdf8; text-decoration: underline;">build.nvidia.com</a> (Generate API Key).</div>
                    </div>
                    """)
                    with gr.Row(elem_id="api-key-input-row"):
                        api_key_input = gr.Textbox(
                            placeholder="Enter NVIDIA API Key (nvapi-...)",
                            value=initial_key_val,
                            type="password",
                            show_label=False,
                            scale=4,
                            elem_id="api-key-input",
                        )
                        btn_connect = gr.Button("🔗 Connect", variant="primary", scale=1, elem_id="btn-connect")
                    api_status_msg = gr.HTML(value=initial_api_status, elem_id="api-status-msg", elem_classes=["tab1-status-pill"])

                with gr.Column(scale=1, elem_classes=["tab1-card"]):
                    gr.HTML("""
                    <div class="tab1-card-header">
                        <div class="tab1-card-title">🤖 Model Architecture (2026 SOTA Hub)</div>
                        <div class="tab1-card-desc">Select an ultra-fast instruction-tuned reasoning model from NVIDIA NIM.</div>
                    </div>
                    """)
                    with gr.Row(elem_id="model-select-row"):
                        model_dropdown = gr.Dropdown(
                            choices=list(AVAILABLE_MODELS.keys()),
                            value=list(AVAILABLE_MODELS.keys())[0],
                            show_label=False,
                            interactive=True,
                            scale=1,
                            elem_id="model-dropdown",
                        )
                    model_status_msg = gr.HTML(value=format_model_status_html(CONFIG["llm_model"]), elem_id="model-status-msg", elem_classes=["tab1-status-pill"])



        # ── TAB 2: UNIFIED KNOWLEDGE INGESTION ─────────────────────────────────
        with gr.TabItem("📁 Upload & Ingest Knowledge"):
            gr.Markdown(
                "### 🌐 Unified Knowledge Hub: Ingest Documents & Website URLs Simultaneously\n"
                "Provide files (**PDF**, **CSV**, **Excel**, **TXT**, **Markdown**) and/or **Website URLs**. "
                "The engine performs layout-aware extraction, tabular serialization, clean web scraping, and constructs a unified **Hybrid FAISS + BM25** search index."
            )
            with gr.Row(equal_height=True, elem_id="tab2-inputs-row"):
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("#### 📤 1. Upload Documents (Multi-Format)")
                    file_upload = gr.File(
                        label="Drag & Drop Files Here (PDF / CSV / XLSX / XLS / TXT / MD)",
                        file_types=[".pdf", ".csv", ".xlsx", ".xls", ".txt", ".md", ".json"],
                        file_count="multiple",
                        height=185,
                        elem_id="file-upload",
                    )
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("#### 🌐 2. Scrape Website URLs (1 or More)")
                    url_input_box = gr.Textbox(
                        label="Enter Website URLs (1 per line)",
                        placeholder="https://en.wikipedia.org/wiki/Retrieval-augmented_generation\nhttps://docs.python.org/3/",
                        lines=4,
                        elem_id="url-input-box",
                    )

            with gr.Row(equal_height=True, elem_id="ingest-action-row"):
                with gr.Column(scale=1, min_width=300):
                    append_toggle = gr.Checkbox(
                        label="Incremental Append Mode (Keep existing knowledge & add new sources)",
                        value=False,
                        elem_id="append-toggle",
                    )
                with gr.Column(scale=1, min_width=300):
                    with gr.Row(elem_id="ingest-btn-subrow"):
                        btn_ingest = gr.Button("⚡ Ingest & Index All Knowledge Sources", variant="primary", scale=3, elem_id="btn-ingest")
                        btn_reset_kb = gr.Button("🗑️ Reset Knowledge Base", variant="secondary", scale=2, elem_id="btn-reset-kb")

            nav_hint = gr.HTML(
                value='<div class="nav-hint-banner">🎉 <b>Knowledge Base Ready!</b> Switch to the <b>💬 Chat with Documents</b> tab to ask cross-source questions in real time.</div>',
                visible=False,
            )

            gr.Markdown("---")
            gr.Markdown("### 🗂️ Active Knowledge Base Inventory")
            manifest_display = gr.HTML(value=render_manifest_table_html(app_state.get("manifest", {})))

            log_output = gr.Textbox(
                label="📋 2026 Pipeline Execution & Telemetry Log",
                lines=8,
                interactive=False,
                elem_id="log-box",
                placeholder="Upload files or enter URLs above and click 'Ingest & Index All Knowledge Sources'...",
            )


        # ── TAB 3: CHAT ───────────────────────────────────────────────────────

        with gr.TabItem("💬 Chat with Documents"):
            with gr.Column():
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    layout="bubble",
                    type="messages",
                    render_markdown=True,
                    show_label=False,
                    elem_id="chatbot-box",
                    placeholder=(
                        "### 👋 Welcome to your 2026 RAG AI Knowledge Assistant!\n\n"
                        "1. **Unified Multi-Source Knowledge**: In the **Upload & Ingest Knowledge** tab, upload **PDF**, **CSV**, **Excel** files and/or enter **Website URLs**.\n"
                        "2. **Cross-Source Q&A**: Ask any question. The assistant automatically queries both your files and scraped websites.\n"
                        "3. **Hybrid Precision & Grounding**: Answers stream in real-time with hybrid BM25 + FAISS citations, similarity scores, and latency stats."
                    ),
                )
                with gr.Row(elem_id="chat-input-row"):
                    msg_input = gr.Textbox(
                        placeholder="Ask a question about the indexed documents or websites...",
                        show_label=False,
                        lines=1,
                        max_lines=3,
                        scale=8,
                        elem_id="msg-input",
                    )
                    btn_send = gr.Button("Send ➤", variant="primary", scale=0, min_width=110, elem_id="btn-send")
                    btn_clear = gr.Button("🗑️", variant="secondary", scale=0, min_width=48, elem_id="btn-clear")

    # ── Event Wiring ──────────────────────────────────────────────────────────

    btn_connect.click(
        fn=cb_configure_api,
        inputs=[api_key_input, model_dropdown],
        outputs=[api_status_msg],
        show_api=False,
    )

    model_dropdown.change(
        fn=cb_update_model,
        inputs=[model_dropdown],
        outputs=[model_status_msg],
        show_api=False,
    )

    btn_ingest.click(
        fn=cb_unified_ingest,
        inputs=[file_upload, url_input_box, append_toggle],
        outputs=[log_output, manifest_display, nav_hint],
        show_api=False,
    )

    btn_reset_kb.click(
        fn=cb_clear_knowledge_base,
        outputs=[log_output, manifest_display, nav_hint],
        show_api=False,
    )

    # Streaming Chat Event
    btn_send.click(
        fn=cb_chat_stream,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input],
        show_api=False,
    )
    msg_input.submit(
        fn=cb_chat_stream,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input],
        show_api=False,
    )

    btn_clear.click(fn=cb_clear_chat, outputs=[chatbot], show_api=False)


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Preload saved index & manifest if available
    saved_idx, saved_chunks, saved_manifest = load_knowledge_base()
    if saved_idx is not None:
        app_state["index"]    = saved_idx
        app_state["chunks"]   = saved_chunks
        app_state["manifest"] = saved_manifest
        print(f"✅ Loaded saved Knowledge Base: {saved_idx.ntotal} vectors, {len(saved_chunks)} chunks, {len(saved_manifest)} sources")

    # Preload API key from environment
    api_key_env = os.environ.get("NVIDIA_API_KEY", "")
    if api_key_env:
        ok, msg = configure_llm(api_key_env)
        print(msg)

    print("\n🚀 Launching RAG AI Knowledge Assistant (2026 World-Class Edition)...")
    print(f"   Model  : {CONFIG['llm_model']}")
    print(f"   URL    : http://localhost:7860\n")

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_api=False,
        inbrowser=True,
    )
