"""FastAPI API routes for Abhinav Digital Twin."""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.pydantic_models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    RoutingInfo,
    SourceInfo,
    ChatMessage,
)
from src.router import route_query
from src.retrievers import retrieve_documents
from src.generator import generate_response, generate_response_stream
from src.query import agentic_rag_with_metadata
from src.utils import get_llm, validate_api_key, get_chroma_client

router = APIRouter()


SAMPLE_QUESTIONS = {
    "Education & Background": [
        "Tell me about your academic journey from BITS Pilani to UW-Madison",
        "What courses did you focus on during your MS in Data Science?",
        "What was campus life like at BITS Pilani?",
    ],
    "Professional Experience": [
        "What did you work on at Flywl?",
        "Describe your role and projects at Asurion",
        "What ML research papers have you published?",
    ],
    "Technical Skills & Projects": [
        "What NLP projects have you worked on?",
        "Tell me about your computer vision experience",
        "What is your tech stack and favorite tools?",
    ],
    "Personal Life": [
        "What is your favorite food and why?",
        "Tell me about growing up in Hyderabad",
        "What books have influenced you the most?",
    ],
}


def _extract_sources(documents) -> list[SourceInfo]:
    """De-duplicate source info from retrieved documents."""
    seen = set()
    sources: list[SourceInfo] = []
    for doc in documents:
        key = (
            doc.metadata.get("source", "Unknown"),
            doc.metadata.get("page", "N/A"),
            doc.metadata.get("category", "N/A"),
        )
        if key not in seen:
            seen.add(key)
            sources.append(
                SourceInfo(filename=key[0], page=key[1], category=key[2])
            )
    return sources


def _run_routing_and_retrieval(query: str):
    """Shared logic: route the query then retrieve documents."""
    validate_api_key()
    router_llm = get_llm(temperature=0.0)
    route_decision = route_query(query, llm=router_llm)

    documents = retrieve_documents(
        query=query,
        collection=route_decision["collection"],
        strategy="semantic",
        k=route_decision["k"],
    )
    return route_decision, documents


# -- Endpoints --

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check with available collections."""
    try:
        client = get_chroma_client()
        collections = [c.name for c in client.list_collections()]
    except Exception:
        collections = []
    return HealthResponse(status="ok", collections=collections)


@router.get("/collections")
def list_collections():
    """List available document collections."""
    try:
        client = get_chroma_client()
        names = [c.name for c in client.list_collections()]
    except Exception:
        names = []
    return {"collections": names}


@router.get("/sample-questions")
def get_sample_questions():
    """Return starter questions grouped by category."""
    return {"questions": SAMPLE_QUESTIONS}


@router.post("/chat")
def chat_endpoint(request: ChatRequest):
    """Main chat endpoint. Supports streaming (SSE) and non-streaming JSON.
    
    Uses LangGraph pipeline with unified agent that has access to both
    RAG documents and calendar tools. The LLM automatically decides when
    to use calendar tools based on the query.
    """
    try:
        # Always use LangGraph pipeline (unified agent with calendar tools)
        history = request.conversation_history if request.conversation_history else None
        result = agentic_rag_with_metadata(
            query=request.query,
            strategy="semantic",
            verbose=False,
            conversation_history=history
        )
        
        # Build routing info
        routing_info = RoutingInfo(
            collection=result.get("collection", ""),
            query_type=result.get("query_type", ""),
            k=result.get("k", 0),
            reasoning=result.get("routing_reasoning", ""),
        )
        
        # Extract sources
        sources = result.get("sources", [])
        
        # For streaming, stream the answer token by token
        if request.stream:
            def event_stream():
                try:
                    # Send routing metadata first
                    meta = {
                        "type": "metadata",
                        "routing": routing_info.model_dump(),
                        "sources": [s for s in sources],
                        "num_documents": result.get("num_documents", 0),
                        "tools_called": result.get("tools_called", []),
                    }
                    yield f"data: {json.dumps(meta)}\n\n"
                    
                    # Stream the answer character by character (simple streaming)
                    answer = result.get("answer", "")
                    for char in answer:
                        yield f"data: {json.dumps({'type': 'token', 'token': char})}\n\n"
                    
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                except Exception as e:
                    import traceback
                    error_msg = f"Error during generation: {str(e)}"
                    yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        
        # Non-streaming path
        return ChatResponse(
            answer=result.get("answer", ""),
            routing=routing_info,
            sources=[SourceInfo(**s) for s in sources],
            num_documents=result.get("num_documents", 0),
        )
            
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
