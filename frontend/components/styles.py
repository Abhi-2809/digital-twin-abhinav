"""Custom CSS for Abhinav Digital Twin Streamlit app."""

import streamlit as st


def apply_custom_css():
    st.markdown(
        """
    <style>
    /* ---------- sidebar text and background ---------- */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #475569;
    }

    /* ---------- sidebar buttons ---------- */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        border-radius: 8px;
        border: 1px solid #475569;
        background: #334155;
        color: #e2e8f0 !important;
        padding: 0.6rem 0.8rem;
        font-size: 0.85rem;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff !important;
        border-color: transparent;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.25);
    }

    /* ---------- sidebar expander ---------- */
    section[data-testid="stSidebar"] details {
        background: #283548;
        border: 1px solid #475569;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] summary {
        color: #e2e8f0 !important;
    }

    /* ---------- chat bubbles ---------- */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin: 0.4rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    /* ---------- header ---------- */
    h1 {
        color: #1e3a8a;
        font-weight: 700;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
