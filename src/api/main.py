"""FastAPI entry point for Abhinav Digital Twin API."""

from dotenv import load_dotenv

# Load env first so LangSmith (LANGCHAIN_*) is set before any LangChain imports
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router

app = FastAPI(
    title="Abhinav Digital Twin API",
    description="Agentic RAG backend for Abhinav's digital twin",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {"message": "Abhinav Digital Twin API", "docs": "/docs"}
