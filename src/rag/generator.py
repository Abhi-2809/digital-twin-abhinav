"""Response generation for Abhinav Digital Twin RAG system."""

from typing import List, Generator, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from src.prompts import RAG_GENERATION_PROMPT
from src.rag.utils import get_llm
from src.data_validation.pydantic_models import ChatMessage


# Conversation context / history handling (inlined from previousConversations.py)
SUMMARY_PROMPT = """Summarize the following conversation history concisely, preserving key facts, topics discussed, and context that would be useful for continuing the conversation. Keep it brief but informative.

Conversation:
{history}

Summary:"""


def build_conversation_context(history: List[ChatMessage]) -> str:
    """Build conversation context from message history within the same session.

    Uses a sliding window approach:
    - Keeps last 3-5 messages in full (recent context)
    - Summarizes older messages if conversation is longer
    - Token-based management (~500 tokens for older messages)

    Args:
        history: List of previous chat messages from current session

    Returns:
        Formatted conversation context string, empty if no history
    """
    if not history:
        return ""

    try:
        # Keep last 4 messages in full (2 user + 2 assistant = good recent context)
        RECENT_MESSAGES_COUNT = 4

        if len(history) <= RECENT_MESSAGES_COUNT:
            # All messages are recent, include them all
            recent_messages = history
            older_summary = ""
        else:
            # Split into recent and older messages
            recent_messages = history[-RECENT_MESSAGES_COUNT:]
            older_messages = history[:-RECENT_MESSAGES_COUNT]
            older_summary = _summarize_older_messages(older_messages)

        # Format recent messages
        recent_text = []
        for msg in recent_messages:
            role = "User" if msg.role == "user" else "Assistant"
            recent_text.append(f"{role}: {msg.content}")

        # Combine summary and recent messages
        parts = []
        if older_summary:
            parts.append(f"Previous conversation summary: {older_summary}")
        parts.append("Recent conversation:")
        parts.extend(recent_text)

        return "\n".join(parts) + "\n"
    except Exception:
        # If summarization fails, just return recent messages
        recent_text = []
        for msg in history[-4:]:
            role = "User" if msg.role == "user" else "Assistant"
            recent_text.append(f"{role}: {msg.content}")
        return "Recent conversation:\n" + "\n".join(recent_text) + "\n"


def _summarize_older_messages(messages: List[ChatMessage]) -> str:
    """Summarize older messages using LLM to stay within token limits.

    If summarization fails or takes too long, returns a simple truncated version.

    Args:
        messages: List of older messages to summarize

    Returns:
        Summary string
    """
    if not messages:
        return ""

    # If only a few messages, just return them truncated (skip LLM call)
    if len(messages) <= 3:
        return " ".join([msg.content[:150] for msg in messages])

    # Format messages for summarization
    history_text = []
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        # Truncate very long messages to avoid token limits
        content = msg.content[:500] if len(msg.content) > 500 else msg.content
        history_text.append(f"{role}: {content}")

    history_str = "\n".join(history_text)

    # Use LLM to summarize older messages
    try:
        llm = get_llm(model="gpt-4.1-nano", temperature=0.0)
        prompt = PromptTemplate(
            input_variables=["history"],
            template=SUMMARY_PROMPT
        )
        response = llm.invoke(prompt.format(history=history_str))
        summary = response.content if hasattr(response, "content") else str(response)
        return summary.strip()
    except Exception:
        # Fallback: return truncated version if summarization fails
        # Don't let summarization errors break the main flow
        return " ".join([msg.content[:100] for msg in messages[-3:]])


def format_documents(documents: List[Document]) -> str:
    """Format retrieved documents into context string."""
    if not documents:
        return "No relevant documents found."

    formatted_docs = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        category = doc.metadata.get("category", "N/A")
        formatted_docs.append(
            f"[{source}] (Page {page}, Category: {category}):\n"
            f"{doc.page_content}\n"
        )
    return "\n---\n".join(formatted_docs)


def format_documents_as_list(documents: List[Document]) -> List[str]:
    """Format retrieved documents as a list of document strings."""
    if not documents:
        return []
    result = []
    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        category = doc.metadata.get("category", "N/A")
        result.append(
            f"[{source}] (Page {page}, Category: {category}):\n{doc.page_content}"
        )
    return result


def _build_prompt(
    documents: List[Document],
    query: str,
    collection: str,
    conversation_history: Optional[List[ChatMessage]] = None
) -> str:
    """Build the formatted RAG prompt string with optional conversation history.
    
    Args:
        documents: Retrieved documents
        query: Current user query
        collection: Collection name
        conversation_history: Previous conversation messages for context
        
    Returns:
        Formatted prompt string
    """
    context = format_documents(documents)
    
    # Build conversation context if history provided (within-session only)
    conversation_summary = ""
    if conversation_history and len(conversation_history) > 0:
        try:
            conversation_summary = build_conversation_context(conversation_history)
        except Exception:
            # If context building fails, continue without it
            conversation_summary = ""
    
    prompt = PromptTemplate(
        input_variables=["collection", "context", "query", "conversation_summary"],
        template=RAG_GENERATION_PROMPT,
    )
    
    # Ensure conversation_summary is not None
    if conversation_summary is None:
        conversation_summary = ""
    
    return prompt.format(
        collection=collection,
        context=context,
        query=query,
        conversation_summary=conversation_summary
    )


def generate_response(
    documents: List[Document],
    query: str,
    collection: str,
    llm: ChatOpenAI = None,
    conversation_history: Optional[List[ChatMessage]] = None,
    model: Optional[str] = None,
) -> str:
    """Generate a complete (non-streamed) response.
    
    Args:
        documents: Retrieved documents
        query: User query
        collection: Collection name
        llm: ChatOpenAI instance
        conversation_history: Previous conversation messages for context
        model: Model name when llm is None (e.g. gpt-4o-mini, GPT-5 nano)
        
    Returns:
        Generated response string
    """
    if llm is None:
        llm = get_llm(model=model or "gpt-4.1-nano", temperature=0.1)

    formatted_prompt = _build_prompt(
        documents, query, collection, conversation_history
    )
    response = llm.invoke(formatted_prompt)
    return response.content if hasattr(response, "content") else str(response)


def generate_response_stream(
    documents: List[Document],
    query: str,
    collection: str,
    llm: ChatOpenAI = None,
    conversation_history: Optional[List[ChatMessage]] = None,
    model: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream response token-by-token using LangChain's stream().
    
    Args:
        documents: Retrieved documents
        query: User query
        collection: Collection name
        llm: ChatOpenAI instance
        conversation_history: Previous conversation messages for context
        model: Model name when llm is None
        
    Yields:
        Response tokens as strings
    """
    if llm is None:
        llm = get_llm(model=model or "gpt-4.1-nano", temperature=0.1)

    try:
        formatted_prompt = _build_prompt(
            documents, query, collection, conversation_history
        )
    except Exception as e:
        yield f"Error building prompt: {str(e)}"
        return

    try:
        for chunk in llm.stream(formatted_prompt):
            if chunk:
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    yield token
    except Exception as e:
        yield f"\n\nError during streaming: {str(e)}"
