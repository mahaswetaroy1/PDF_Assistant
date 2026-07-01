# PDF Assistant — RAG Q&A App

A Retrieval-Augmented Generation (RAG) application that lets you upload any PDF and ask questions about it. Answers are grounded only in the content of the uploaded document — no hallucinated information from the model's general training data.

**Live app:** https://docu-assist.streamlit.app/

---

## Why I built this

RAG is one of the most talked-about concepts in AI right now, but a lot of tutorials rely on libraries like LangChain that abstract away what's actually happening. I built this from scratch without LangChain to understand every step of the pipeline: chunking, embeddings, similarity search, and grounded prompting.

---

## What it does

- Upload any PDF
- Ask questions about its contents in natural language
- Get answers generated using only the relevant sections of your document
- View the exact source chunks used to generate each answer
- Maintains conversation history within a session

---

## How it works

1. **Extract** — Text is extracted from the uploaded PDF using `pypdf`
2. **Chunk** — Text is split into overlapping chunks to preserve context across boundaries
3. **Embed** — Each chunk is converted into a vector embedding using `sentence-transformers` (`all-MiniLM-L6-v2`)
4. **Store** — Embeddings are kept in memory as NumPy arrays for the session
5. **Retrieve** — When a question is asked, it's embedded the same way, and cosine similarity is used to find the most relevant chunks
6. **Generate** — The top matching chunks are passed to Claude (Anthropic API) with an instruction to answer only from the provided context
7. **Display** — The answer is shown along with the exact source chunks used, so the response is verifiable

---

## Tech stack

- **Python**
- **Streamlit** — web interface
- **pypdf** — PDF text extraction
- **sentence-transformers** — text embeddings
- **NumPy** — cosine similarity calculations
- **Anthropic API** — response generation (Claude)
- **python-dotenv** — local environment variable management

---

## Running it locally

```bash
git clone https://github.com/mahaswetaroy1/PDF_Assistant.git
cd PDF_Assistant
pip3 install -r requirements.txt
```

Create a `.env` file in the project root with your Anthropic API key:

```
ANTHROPIC_API_KEY=your_key_here
```

Then run:

```bash
python3 -m streamlit run app.py
```

---

## Notes on design decisions

- **Chunking with overlap** — prevents important context from being lost when a chunk boundary falls in the middle of a relevant sentence
- **Grounded prompting** — the model is explicitly instructed to answer only from retrieved context and to say so if the answer isn't present, reducing hallucination
- **Session-based history** — chat history and sources are stored per-session using Streamlit's `session_state`, so previous Q&A pairs remain visible with their original sources

---

## Possible future improvements

- Support for multiple documents in a single session
- Persistent vector storage (e.g. a vector database) instead of in-memory embeddings
- OCR support for scanned/image-based PDFs
- Adjustable number of retrieved chunks per query