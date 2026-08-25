---
title: RAG AI Knowledge Assistant
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: mit
short_description: Multi-Modal Hybrid RAG Knowledge Assistant with NVIDIA NIM
---

# 🤖 RAG AI Knowledge Assistant (2026 Production Edition)

> An enterprise-grade, multi-modal **Retrieval-Augmented Generation (RAG)** pipeline designed for multi-format document ingestion (PDF, CSV, Excel, TXT, Markdown) and live web scraping with **Hybrid Search (FAISS + BM25)**, **Zero-Hallucination Guardrails**, and real-time streaming citations powered by **NVIDIA NIM LLMs**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/dhananiyash9/rag-ai-knowledge-assistant)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yashdhanani/rag-ai-knowledge-assistant)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA-NIM%20API-76b900.svg)](https://build.nvidia.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS%20Dense-purple.svg)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/Lexical%20Search-BM25%20Rank-orange.svg)](https://github.com/dorianbrown/rank_bm25)
[![Gradio](https://img.shields.io/badge/UI-Gradio%205-FF7C00.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

### 🌐 Live Interactive Demo
🚀 **Try the live app directly in your browser:**  
👉 **[https://huggingface.co/spaces/dhananiyash9/rag-ai-knowledge-assistant](https://huggingface.co/spaces/dhananiyash9/rag-ai-knowledge-assistant)**  
*(Bring Your Own Free NVIDIA API Key & Test with your documents!)*

---

## 📋 Overview

Standard Large Language Models often suffer from hallucinations, outdated knowledge, and lack of verifiable citations. This project implements a **state-of-the-art Hybrid RAG architecture** that combines dense semantic retrieval with sparse lexical keyword matching to provide accurate, grounded answers directly from your private knowledge base.

### 🌟 Key Capabilities
- **Multi-Format Document Ingestion**: Native parsers for `PDF`, `CSV`, `Excel (.xlsx)`, `TXT`, and `Markdown (.md)`.
- **Live Web Ingestion**: Scrape and index dynamic website content from URLs with real-time DOM cleaning and tokenization.
- **Hybrid Search Engine**: Fuses **FAISS dense cosine vectors** (`all-MiniLM-L6-v2`) and **BM25 lexical scoring** for high precision retrieval.
- **Zero-Hallucination Guard**: Strict grounding prompt constraints and similarity thresholds to reject out-of-context queries.
- **Multi-Model Selector**: Switch on-the-fly between **Meta Llama-3.1-8B-Instruct**, **Llama-3.3-70B**, **Nemotron**, **Mistral**, and **Gemma**.
- **Dynamic Dark / Light Theme**: Apple-grade glassmorphic interface with full dark/light synchronization.
- **Full Source Citations & Telemetry**: Every answer includes document/URL source badges, page numbers, similarity scores, and retrieval latency metrics.

---

## 🏗️ Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MULTI-SOURCE INGESTION ENGINE                         │
│                                                                             │
│  📁 PDF / CSV / Excel / TXT / MD   ─┐                                       │
│                                      ├─▶ Adaptive Chunking ─▶ Embeddings    │
│  🌐 Live Website URLs (Scraper)    ─┘   (Format-Aware)       (MiniLM-L6-v2) │
│                                                                   │         │
│                                                                   ▼         │
│                                              🗄️ Hybrid Vector Store         │
│                                              • FAISS Dense Index            │
│                                              • BM25 Lexical Inverted Index  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ONLINE QUERY & REASONING                           │
│                                                                             │
│  ❓ User Question ─▶ Query Embedding ─▶ Hybrid Retrieval (FAISS + BM25)     │
│                                                    │                        │
│                                                    ▼                        │
│                                         🛡️ Context Guard & Grounding        │
│                                                    │                        │
│                                                    ▼                        │
│                                        🤖 NVIDIA NIM Cloud Engine           │
│                                           (Llama 3.1 / Nemotron)            │
│                                                    │                        │
│                                                    ▼                        │
│                                         ⚡ Streaming Response               │
│                                         📌 Grounded Source Citations        │
│                                         ⏱️ Latency & Similarity Metrics     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|---|---|---|
| **Embeddings** | `sentence-transformers` | Dense vectors via `all-MiniLM-L6-v2` (384 dims) |
| **Vector Database** | `faiss-cpu` | Inner product cosine similarity vector indexing |
| **Lexical Engine** | `rank-bm25` | Okapi BM25 keyword matching for hybrid retrieval |
| **Document Parsers** | `PyMuPDF`, `openpyxl`, `pandas` | High-fidelity extraction for PDF, Excel, CSV, text |
| **Web Scraper** | `beautifulsoup4`, `requests` | Clean DOM text extraction and metadata enrichment |
| **LLM Inference** | `OpenAI SDK` / NVIDIA NIM | OpenAI-compatible endpoint for NVIDIA Cloud models |
| **Frontend UI** | `Gradio 5` | Reactive interface with custom CSS & dynamic themes |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10 or higher installed.
- An **NVIDIA NIM API Key** (Free tier available at [build.nvidia.com](https://build.nvidia.com/)).

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/rag-knowledge-assistant.git
cd rag-knowledge-assistant
```

### 3. Set Up a Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Application
```bash
# Option A: Export your API key in environment
export NVIDIA_API_KEY="your-nvapi-key-here"
python3 app.py

# Option B: Run and enter API key in the Setup UI
python3 app.py
```

### 6. Access the Web Interface
Open your web browser and navigate to:
```
http://localhost:7860
```

---

## 💡 How to Use

1. **Tab 1: ⚙️ Setup & Model Selector**
   - Verify your NVIDIA API key connection status.
   - Choose your preferred model (e.g., `Meta Llama-3.1-8B-Instruct` or `Llama-3.3-70B-Instruct`).
2. **Tab 2: 📥 Upload & Ingest Knowledge**
   - Upload any document (`.pdf`, `.csv`, `.xlsx`, `.txt`, `.md`) or paste website URLs.
   - Toggle **Incremental Append Mode** to add new documents without wiping existing indexes.
   - Click **Ingest & Index All Knowledge Sources**.
3. **Tab 3: 💬 Chat with Documents**
   - Ask natural language questions in real-time.
   - Inspect grounded answers, verifiable source citations, page numbers, and similarity metrics.

---

## 📁 Project Structure

```
RAG-Based AI Knowledge Assistant/
├── app.py                             # Complete standalone Gradio application
├── RAG_AI_Knowledge_Assistant.ipynb   # Complete interactive Jupyter Notebook
├── sample_company_data.csv            # Sample CSV dataset for testing
├── requirements.txt                    # Python package dependencies
├── .gitignore                         # Git ignore rules for clean commits
└── README.md                           # Project documentation & guide
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
