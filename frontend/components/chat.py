"""Chat components for Abhinav Digital Twin Streamlit app."""

import json
import streamlit as st
import httpx
from frontend.config import API_BASE_URL

# Friendly display names for internal tool function names
TOOL_DISPLAY_NAMES = {
    "get_google_calendar_events": "Google Calendar",
    "create_google_calendar_event": "Google Calendar",
    "get_daily_news_summary": "Daily News Agent",
}


def render_chat_history():
    """Render all messages stored in session state."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # Show tool indicator if present
            if msg.get("tools_called"):
                _render_tool_indicator(msg["tools_called"])
            st.markdown(msg["content"])


def _render_tool_indicator(tools_called: list[str]):
    """Render a styled indicator showing which tools were called."""
    for tool_name in tools_called:
        display_name = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
        st.markdown(
            f'<div style="display:inline-flex;align-items:center;gap:6px;'
            f'background:#f0f2f6;border-radius:8px;padding:6px 14px;'
            f'margin-bottom:8px;font-size:0.85rem;color:#555;">'
            f'<span style="font-size:1rem;">&#128295;</span> '
            f'Used tool: <strong>{display_name}</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )


def stream_chat_response(query: str, conversation_history: list = None):
    """Call the FastAPI streaming endpoint and yield tokens.
    
    Yields special dicts for metadata (including tools_called) and
    plain strings for answer tokens.

    Args:
        query: Current user query
        conversation_history: Previous conversation messages for context (should exclude current query)
    """
    if conversation_history is None:
        conversation_history = []
    
    # Format history for API (exclude current query if it's in the history)
    history_for_api = []
    for msg in conversation_history:
        # Skip if this message matches the current query (it's the message we're responding to)
        if msg.get("content") != query:
            history_for_api.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
    
    try:
        with httpx.stream(
            "POST",
            f"{API_BASE_URL}/chat",
            json={
                "query": query,
                "stream": True,
                "conversation_history": history_for_api
            },
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                
                try:
                    payload = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                if payload.get("type") == "metadata":
                    # Forward tool info to the caller
                    tools = payload.get("tools_called", [])
                    if tools:
                        yield {"type": "tools_called", "tools": tools}
                elif payload.get("type") == "token":
                    yield payload.get("token", "")
                elif payload.get("type") == "done":
                    break
                elif payload.get("type") == "error":
                    yield f"\n\nError: {payload.get('message', 'Unknown error')}"
                    break

    except httpx.ConnectError:
        yield (
            "Could not connect to the backend. "
            "Please make sure the FastAPI server is running on port 8000."
        )
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        reason = e.response.reason_phrase or "HTTP error"
        yield f"HTTP Error {status}: {reason}"
    except Exception as exc:
        yield f"Error: {str(exc)}"
