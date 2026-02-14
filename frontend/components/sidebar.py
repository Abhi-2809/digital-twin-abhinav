"""Sidebar component with sample questions."""

import streamlit as st
import httpx
from frontend.config import API_BASE_URL


def _fetch_sample_questions() -> dict:
    """Pull sample questions from the backend, with a local fallback."""
    try:
        resp = httpx.get(f"{API_BASE_URL}/sample-questions", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("questions", {})
    except Exception:
        pass

    # Fallback if backend is unreachable
    return {
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


def render_sidebar():
    """Render the sidebar with sample questions and controls."""
    with st.sidebar:
        st.markdown("### Chat with Abhinav's Digital Twin")
        st.markdown("Hi I am Abhinav's digital twin , Ask me anything about my personal or professional interests")
        st.divider()

        st.markdown("#### Sample Questions")

        questions = _fetch_sample_questions()
        for category, items in questions.items():
            with st.expander(category, expanded=False):
                for q in items:
                    if st.button(q, key=f"sq_{q[:30]}"):
                        st.session_state.pending_question = q
                        st.rerun()

        st.divider()

        # Clear chat at the bottom
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
