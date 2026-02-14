"""Streamlit frontend for Abhinav Digital Twin."""

import sys
from pathlib import Path

# Make project root importable so `frontend.*` and `src.*` resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from frontend.components.styles import apply_custom_css
from frontend.components.sidebar import render_sidebar
from frontend.components.chat import render_chat_history, stream_chat_response, _render_tool_indicator

# -- Page config --
st.set_page_config(
    page_title="Chat with Abhinav",
    page_icon="https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f4ac.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()

# -- Session state --
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# -- Sidebar --
render_sidebar()

# -- Header --
st.title("Chat with Abhinav's Digital Twin")
st.markdown("Hi I am Abhinav's digital twin , Ask me anything about my personal or professional interests")

# -- Chat history --
render_chat_history()

# -- Welcome message --
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Hey! I'm Abhinav's digital twin. "
            "You can ask me about my education, work experience, research, "
            "personal interests, or anything else. "
            "Try one of the sample questions in the sidebar to get started!"
        )

# -- Always render chat input so it never disappears --
typed_input = st.chat_input("Ask me anything...")

# -- Pick whichever source fired: sidebar click OR typed text --
user_input = None
if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None
elif typed_input:
    user_input = typed_input

# -- Process query --
# Don't append to session state until after we have the response, so history
# doesn't include the new user message when we render it, avoiding duplicate
# and wrong order (previous answer showing below new question).
if user_input:
    # Show new user message (not from history yet)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream new assistant response (no keyed container so Streamlit doesn't reuse old content)
    with st.chat_message("assistant"):
        tool_container = st.empty()
        placeholder = st.empty()
        full_response = ""
        tools_called = []
        history = list(st.session_state.messages)  # history does not include current user_input

        try:
            for chunk in stream_chat_response(user_input, conversation_history=history):
                if isinstance(chunk, dict) and chunk.get("type") == "tools_called":
                    tools_called = chunk["tools"]
                    with tool_container.container():
                        _render_tool_indicator(tools_called)
                elif isinstance(chunk, str) and chunk:
                    full_response += chunk
                    placeholder.markdown(full_response + " |")
        except Exception as e:
            full_response += f"\n\nError: {str(e)}"

        placeholder.markdown(full_response)

    # Append both messages only after response is complete so next run shows them from history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response, "tools_called": tools_called}
    )
    st.rerun()
