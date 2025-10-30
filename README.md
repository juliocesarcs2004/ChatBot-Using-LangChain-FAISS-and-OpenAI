# Chatbot PDF — Streamlit App with OpenAI and LangChain

This repository contains a simple Streamlit application that lets you upload a PDF, index its content with embeddings, and use an OpenAI model (via LangChain) to answer questions based on the document's text.

Quick summary
- PDF upload ➜ text extraction (PyPDF2)
- Split text into "chunks" (RecursiveCharacterTextSplitter)
- Generate embeddings (OpenAIEmbeddings / LangChain)
- Index with FAISS
- Similarity search and QA chain (load_qa_chain + ChatOpenAI)
- Display only the answer to the user, with an optional expander to view source passages

Main file
- `chatbot.py`: Streamlit app that implements the full flow (upload, processing, search and answer).

Prerequisites
- macOS / Linux / Windows with Python 3.10+ (3.11+ recommended)
- An OpenAI account and an API key (OPENAI_API_KEY)

Recommended dependencies
You can install dependencies with pip. Below is a suggested list; some packages (e.g., FAISS) may require special installation steps (conda) on macOS.

Example basic installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install streamlit python-dotenv PyPDF2 langchain langchain-openai langchain-community faiss-cpu openai
```

Notes about FAISS on macOS
- `faiss-cpu` installed via pip does not always work on macOS. If the pip install fails, I recommend using conda/miniforge:

```bash
conda create -n chatbot python=3.10
conda activate chatbot
conda install -c conda-forge faiss-cpu
pip install streamlit python-dotenv PyPDF2 langchain langchain-openai langchain-community openai
```

Environment variables
- Create a `.env` file in the project root with the following content:

```env
OPENAI_API_KEY=sk-...your_key_here...
```

- Alternatively, export the variable in your terminal before running:

```bash
export OPENAI_API_KEY="sk-...your_key_here..."
```

Important note about libomp / OpenMP (OMP: Error #15)
- On some Macs you may encounter this message and the application may abort because libomp was initialized more than once. The `chatbot.py` file sets the environment variable `KMP_DUPLICATE_LIB_OK=TRUE` at the top of the script as a pragmatic workaround to avoid the abort caused by multiple copies of the OpenMP runtime.
- This is a pragmatic, unsupported workaround — the ideal fix is to ensure only a single OpenMP runtime is linked in your Python environment (reinstall conflicting packages, use conda, etc.).
- If you prefer not to use the workaround, remove the line that sets `KMP_DUPLICATE_LIB_OK` and resolve the package conflicts.

Running the app

```bash
# activate venv (if created)
source .venv/bin/activate

# run Streamlit
streamlit run Chatbot.py
```

Usage
1. Open the URL that Streamlit prints in the terminal (usually http://localhost:8501).
2. Use the sidebar to upload a PDF file.
3. After upload and indexing, type your question into the "Type your question here" field.
4. The app will display only the generated answer. To inspect the source passages (the chunks), click "Show source passages (raw chunks)".

How `chatbot.py` works (detailed view)
- Text extraction: uses `PyPDF2.PdfReader` to extract text from pages. PDFs composed only of images (scans) may not yield extractable text.
- Chunking: `RecursiveCharacterTextSplitter` with chunk_size=1000 and chunk_overlap=150 (adjust as needed).
- Embeddings: tries to import `OpenAIEmbeddings` from the `langchain-openai` package (recommended) and falls back to `langchain_community.embeddings` if needed.
- Indexing: uses FAISS (`FAISS.from_texts`) to create a local vector store.
- Search: `vector_store.similarity_search(question)` returns matching documents.
- LLM/QA: uses `ChatOpenAI` through LangChain (if available) and `load_qa_chain(..., chain_type='stuff')` to generate the answer based on the returned documents.
- Display: only the generated answer is shown in the UI; source passages are available inside an optional expander for inspection.

Common troubleshooting
- libomp / OMP: Error #15: see the section above about `KMP_DUPLICATE_LIB_OK`. Ideally fix conflicting dependencies rather than relying on the workaround.
- `OpenAIEmbeddings` deprecation warning: depending on LangChain's version, you may see a warning that `OpenAIEmbeddings` moved to the `langchain-openai` package. Install or upgrade `langchain-openai` as recommended:

```bash
pip install -U langchain-openai
```

- FAISS install issues: use conda/conda-forge for a more reliable installation on macOS/Windows.

Suggested improvements (next steps)
- Persist the FAISS index to disk between runs to avoid recomputing embeddings on every upload (use `faiss.write_index` / `FAISS.save_local`).
- Support scanned PDFs by integrating OCR (Tesseract + pytesseract) before text extraction.
- UI/UX: show indexing progress, allow multiple documents, add a Q&A history.
- Security: do not log your API keys; keep them in secure environment variables.
