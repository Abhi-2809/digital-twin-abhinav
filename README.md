# Abhinav's Digital Twin - Question-Answering Over Your Documents with Citations and Agentic Tools

This project builds a **digital twin** that answers questions as you would, using your own personal and professional documents. Responses are **grounded in your documents** and **cite sources** so you can see where each claim comes from. The twin has **agentic capabilities** via tools such as Google Calendar (list, create, delete events), Gmail (send email), and a daily news summarizer based on custom topics and sources. Response quality is **evaluated using the RAGAS framework** (faithfulness, answer relevancy, noise sensitivity).

---

## Tech stack

- **Backend:** Python 3, FastAPI, Uvicorn, SSE (Server-Sent Events)
- **Frontend:** Streamlit
- **RAG:** LangChain, LangChain-OpenAI, ChromaDB, OpenAI Embeddings, PyPDF
- **Agent:** LangGraph, LangChain tools
- **Integrations:** Google API Python Client, Google Auth (OAuth2) for Calendar and Gmail
- **Evaluation:** RAGAS (NoiseSensitivity, AnswerRelevancy, Faithfulness, Latency, Cost)
- **Others:** python-dotenv, Pydantic, BeautifulSoup, httpx

---

## Project structure

```
digital-twin-viven/
├── frontend/
│   ├── app.py                 # Streamlit entrypoint
│   ├── config.py
│   └── components/
│       ├── chat.py
│       ├── sidebar.py
│       └── styles.py
├── src/
│   ├── api/
│   │   ├── main.py            # FastAPI app
│   │   └── routes.py          # /api/health, /api/chat (streaming + non-streaming)
│   ├── rag/
│   │   ├── ingest.py          # Load PDFs from data/ → Chroma collections
│   │   ├── retrievers.py      # Personal / professional retrieval
│   │   ├── router.py          # Query routing (personal vs professional, k)
│   │   ├── generator.py       # Answer generation with citations
│   │   └── utils.py           # LLM/embedding helpers
│   ├── langgraph_workflow/
│   │   └── workflow.py       # Agent graph (RAG + tools)
│   ├── tools/
│   │   ├── tools.py           # Calendar, email, news tools
│   │   └── google_clients/
│   │       ├── calendar_client.py
│   │       └── email_client.py
│   ├── data_validation/
│   │   └── pydantic_models.py
│   └── prompts.py
├── data/
│   ├── personal/              # PDFs for “personal” collection
│   └── professional/         # PDFs for “professional” collection
├── vector_store/              # ChromaDB persistence (created by ingest)
├── Testing/
│   ├── ragas_results.py       # RAGAS evaluation script
│   ├── test_queries.json
│   └── ...
├── requirements.txt
└── README.md
```

---

## Run from scratch

### 1. Clone and setup

```bash
cd digital-twin-viven
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API keys and credentials

- **OpenAI:** Set `OPENAI_API_KEY` in the environment or in a `.env` file in the project root.
- **Google Calendar / Gmail:**  
  - Create OAuth2 credentials in Google Cloud Console (Desktop app).  
  - Download the JSON and set `GOOGLE_CALENDAR_CREDENTIALS_PATH` to the path of that JSON file.  
  - On first run, complete the OAuth flow; a `token.json` will be created.  
  (See `GOOGLE_CLOUD_SETUP.md` in the repo for step-by-step setup.)


### 3. Ingest documents

Place PDFs in `data/personal/` and/or `data/professional/`, then run:

```bash
PYTHONPATH=. python -m src.rag.ingest
```

This creates/updates Chroma collections under `vector_store/`.

### 4. Start backend

```bash
PYTHONPATH=. uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Start frontend

In another terminal (with the same venv activated):

```bash
PYTHONPATH=. streamlit run frontend/app.py --server.port 8501
```

Open `http://localhost:8501` to chat with the digital twin.

### 6. (Optional) RAGAS evaluation

```bash
PYTHONPATH=. python Testing/ragas_results.py
```

Requires `OPENAI_API_KEY`. Results are written under `Testing/` (e.g. `ragas_results.json`, `ragas_results_output.txt`).

---

## Assumptions

- Designed as a **personal tool** , not for a company setting
- Users can be **the builder (me) or external users** — e.g. to look up information about my life, ask about events, or send me an email.
- Queries can be fact-based or more comprehensive about me
- Any questions that are not about me will not be answered and redirected back to about me

---

## Architecture

A user **query** first hits the **agent** (LangGraph). The agent decides whether the query **needs a tool** (e.g. calendar, email, news). If yes, the corresponding **tool is invoked and executed**, and the tool output is turned into a response. If no, the query goes into the **RAG pipeline**: a **router** classifies the query (personal vs professional collection, and fact-based vs comprehensive), which sets **top-k** (e.g. fewer chunks for fact-based, more for comprehensive). The **retriever** fetches relevant chunks from the chosen collection(s), and the **generator** produces an answer grounded in those chunks and returns it with **citations**.

**Flow:**

```
                    ┌─────────────┐
                    │ User Query  │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Agent     │
                    │ (LangGraph) │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
     Needs tool?                   No tool
              │                         │
              ▼                         ▼
       ┌─────────────┐           ┌─────────────────────┐
       │ Call tool   │           │ Router (db, top k)  │
       │ (Calendar,  │           │ personal/professional│
       │  Email,     │           │ fact vs comprehensive│
       │  News)      │           │ → top-k             │
       └──────┬──────┘           └──────┬──────────────┘
       ┌─────────────┐                   │
       │ Execute     │                   ▼
       │ tool        │           ┌─────────────┐
       └──────┬──────┘           │ Retriever   │
              │                  │ (ChromaDB)  │
              │                  └──────┬──────┘
              │                          │
              │                          ▼
              │                  ┌─────────────┐
              │                  │ Generator   │
              │                  │ (context +  │
              │                  │  citations) │
              │                  └──────┬──────┘
              │                          │
              └────────────┬─────────────┘
                           ▼
                    ┌─────────────┐
                    │  Response   │
                    └─────────────┘
```

---

## System Design choices

**Vector store**

- **ChromaDB** — lighter, in-built metadata filtering (e.g. by `category` for personal vs professional).
- *Others considered:* Faiss (no in-built metadata filtering), Qdrant (heavier for this use case).

**Embedding**

- **text-embedding-3-small** — cost-effective, as suggested in the assignment.

**Chunking**

- **RecursiveCharacterTextSplitter** — preserves document structure and keeps paragraphs/sentences together.
- *Others considered:* Keyword-based chunking (less suited to structured content like research papers); semantic chunking (better for long-form with topic shifts, but more compute-heavy).

**Retriever**

- Used an ANN (approximate nearest neighbor) vector search semantic retriever to   retrieve relevant chunks. For this use case, semantic retrieval was preferred to ensure context-aware answers. However, hybrid retrieval (combining dense and keyword-based methods) could also be applied and may further improve retrieval quality; we should evaluate this using retrieval metrics such as percison recall and MRR.



**Evaluation**

- Used a set of **13 test queries** (curated by me), run against the pipeline and scored with RAGAS.
- Evaluated using only fact-based queries for now (Noise Sensitivity was high along with the correct answer other extra information was also being generated so i modified the prompt to make it strictly answer to the point later)

| Model         | Noise sensitivity | Answer relevancy | Faithfulness | Avg time (s) | Avg cost   |
|---------------|-------------------|------------------|--------------|---------------|------------|
| gpt-4o-mini   | 0.376             | 0.454            | 0.614        | 2.33          | 0.000213   |
| **gpt-4.1-nano** | **0.379**      | **0.632**        | **0.765**    | **0.81**      | **0.000139** |
| gpt-4.1-mini  | 0.398             | 0.570            | 0.786        | 1.54          | 0.000569   |

**GPT-4.1-nano** performed best on answer relevancy and faithfulness while having the lowest latency and cost, so we use it for generation.


**Generation**

I evaluated three models on RAGAS metrics (noise sensitivity, answer relevancy, faithfulness) plus latency and cost, and chose the best-performing one for generation. 

Note: During generation we pass the **previous 4 messages** and a **conversation summary** for context.

---

## Future scope

- Evaluation for retrieval (e.g. retrieval-level metrics).
- Handle more complex / multi-hop queries (multiple sub-questions in one query).
- Re-ranking of retrieved results (especially with larger document sets).
- Tool-calling evaluation.
