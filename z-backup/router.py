"""Query routing and classification for Abhinav Digital Twin"""

import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from src.prompts import ROUTER_PROMPT
from src.utils import get_llm
from src.pydantic_models import RouterOutput


def route_query(query: str, llm: ChatOpenAI = None) -> Dict[str, Any]:
    """Route query to appropriate collection and determine retrieval strategy.
    
    Uses LLM to classify:
    1. Collection: personal, professional, or both
    2. Type: fact-based or comprehensive
    3. k: number of documents to retrieve (3 for facts, 10 for comprehensive)
    
    Args:
        query: User query string
        llm: ChatOpenAI instance (creates new if None)
        
    Returns:
        Dictionary with keys: collection, type, k, reasoning (validated via RouterOutput)
    """
    if llm is None:
        llm = get_llm(temperature=0.0)
    
    prompt = PromptTemplate(
        input_variables=["query"],
        template=ROUTER_PROMPT
    )
    formatted_prompt = prompt.format(query=query)
    response = llm.invoke(formatted_prompt)
    content = response.content if hasattr(response, 'content') else str(response)
    
    try:
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        raw = json.loads(content)
        validated = RouterOutput.from_llm_response(raw)
        return validated.model_dump()
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Failed to parse routing decision: {e}")
        print(f"Response content: {content}")
        fallback = RouterOutput(
            collection="both",
            type="comprehensive",
            k=10,
            reasoning="Default routing due to parsing error"
        )
        return fallback.model_dump()


def print_routing_decision(route: Dict[str, Any], query: str) -> None:
    """Print formatted routing decision.
    
    Args:
        route: Routing decision dictionary
        query: Original query string
    """
    print("\n" + "="*80)
    print("QUERY ROUTING DECISION")
    print("="*80)
    print(f"Query: {query}")
    print(f"Collection: {route['collection']}")
    print(f"Type: {route['type']}")
    print(f"Top-K: {route['k']}")
    print(f"Reasoning: {route.get('reasoning', 'N/A')}")
    print("="*80 + "\n")
