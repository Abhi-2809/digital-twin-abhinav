First assumption is that I am only including two categories of my life for now: personal and professional, and I have data sources related to that. 
Second assumption is that the user can ask both fact-based and summary types of queries 



viven-digital-twin/
├── data/                          # Your existing PDFs ✅
│   ├── personal/
│   │   ├── personal_bio.pdf
│   │   └── book_reviews.pdf
│   └── professional/
│       ├── resume-abhinav-ae.pdf
│       ├── professional_summary.pdf
│       └── ... (your other PDFs)
|
├── src/                           # NEW: Modular Python code
│   ├── __init__.py
│   ├── ingest.py                  # Chroma collections (from prev prompt)
│   ├── query.py                   # RAG retrieval + top-k logic
│   ├── rag.py                     # LLM chain (OpenAI mini)
│   └── agents/                    # FUTURE: Agentic RAG
│       └── __init__.py
|
├── vector_store/                  # Chroma output (gitignored)
|
├── app.py                         # Streamlit frontend
├── requirements.txt               # Dependencies
├── .env                           # API keys (gitignored)
├── .gitignore                     # vector_store/, .env
└── README.md                      # Viven submission



System Design choices


Vector Store:

Cromadb (lighter, in built meta data filtering)
Others:
Faiss(no in build meta data filtering)
Qdrant(heavier use)

Embedding generation:

text-embedding-3-small (cost effective as mentioned in assignment)

Chunking:

RecursiveCharacterTextSplitter - Preserves document structure and groups paragrahs/sentences together
Others:
Keyword chunking - Not exactly strucutred data like research papers..
Semantic chunking - useful for long form documents when there are topic shifts and more comp expensive


Generation: 

Model check for evaluation metrics and then decide between 
the three whichever is performing the best

Evaluation:

