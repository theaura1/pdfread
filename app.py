# app.py

import os
import streamlit as st
from rag_engine import load_and_prepare_docs, build_qa_chain

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📄 Ask Your PDF (OCI GenAI)",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── GLOBAL DARK-THEME + ANIMATIONS ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Import a crisp sans-serif */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    /* Animate the page background through multiple shades */
    [data-testid="stAppViewContainer"] {
      font-family: 'Inter', sans-serif;
      background: linear-gradient(-45deg, #1a1a2e, #0f0f1e, #1f1f39, #0f0f1e);
      background-size: 400% 400%;
      animation: gradientBG 20s ease infinite;
      color: #e0e0e0;
    }
    @keyframes gradientBG {
      0%   { background-position: 0%   50%; }
      50%  { background-position: 100% 50%; }
      100% { background-position: 0%   50%; }
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
      background: rgba(26, 26, 46, 0.95);
      border-right: 1px solid #333;
    }

    /* Constrain the main container width */
    .block-container {
      max-width: 850px;
      padding: 2rem 1rem;
    }

    /* Gradient-clipped title */
    .stMarkdown h1 {
      font-size: 3rem;
      font-weight: 800;
      text-align: center;
      margin-bottom: 0.3rem;
      background: linear-gradient(90deg, #ff6ec4, #7373ff);
      -webkit-background-clip: text;
      color: transparent;
    }

    /* ANSWER BOX: rotating gradient BORDER, no overlay */
    .answer-box {
      position: relative;
      background: rgba(42,42,63,0.8);
      border: 4px solid transparent;
      border-radius: 12px;
      padding: 1.5rem;
      box-shadow: 0 4px 30px rgba(0,0,0,0.5);
      animation: fadeIn 0.6s ease-out, rotateBorder 8s linear infinite;
      --bdeg: 0deg;
      border-image: linear-gradient(var(--bdeg), #ff6ec4, #7373ff, #18dcff) 1;
      color: #f5f5f5;
      margin-bottom: 1rem;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(-10px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes rotateBorder {
      to { --bdeg: 360deg; }
    }

    /* Text input styling */
    .stTextInput > label {
      color: #ddd !important;
    }
    .stTextInput > div > input {
      background: rgba(42,42,63,0.7) !important;
      border: 1px solid #555 !important;
      border-radius: 8px !important;
      color: #e0e0e0 !important;
      padding: 0.5rem 1rem !important;
    }

    /* Button styling */
    .stButton > button {
      background: linear-gradient(90deg, #ff6ec4, #7373ff) !important;
      border: none !important;
      border-radius: 8px !important;
      padding: 0.6rem 1.2rem !important;
      font-weight: 600 !important;
      box-shadow: 0 4px 15px rgba(0,0,0,0.4) !important;
      transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stButton > button:hover {
      transform: translateY(-2px) !important;
      box-shadow: 0 6px 20px rgba(0,0,0,0.6) !important;
    }

    /* Expander styling */
    .stExpanderHeader {
      background: rgba(42,42,63,0.7) !important;
      border: 1px solid #444 !important;
      border-radius: 8px !important;
      padding: 0.5rem 1rem !important;
      color: #ddd !important;
    }
    .stExpanderHeader:hover {
      background: rgba(42,42,63,0.9) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── SIDEBAR ─────────────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Controls")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf", help="Max 200 MB")
# use_cache     = st.sidebar.checkbox("Cache docs & chain", value=True, help="Speeds up repeated runs")
st.sidebar.markdown("---")
st.sidebar.write("Built with OCI GenAI by Lavkesh")

# ── MAIN CONTENT ─────────────────────────────────────────────────────────────────
st.markdown("# 📄 Ask Your PDF")
st.write("Upload a PDF in the sidebar, then ask any question about its contents.")

if uploaded_file:
    # Save locally
    os.makedirs("data", exist_ok=True)
    pdf_path = os.path.join("data", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.success(f"✅ `{uploaded_file.name}` uploaded!")

    # Load & split
    if use_cache:
        @st.cache_data(ttl=3600, show_spinner=False)
        def _prepare(path): return load_and_prepare_docs(path)
        docs = _prepare(pdf_path)
    else:
        docs = load_and_prepare_docs(pdf_path)

    # Build or reuse chain
    if use_cache:
        @st.cache_resource(show_spinner=False)
        def _chain(docs): return build_qa_chain(docs)
        qa = _chain(docs)
    else:
        qa = build_qa_chain(docs)

    # Question & display
    query = st.text_input("💬 Your question:")
    if query:
        with st.spinner("Thinking…"):
            res = qa(query)

        st.markdown("### 📝 Answer")
        st.markdown(f'<div class="answer-box">{res["result"]}</div>', unsafe_allow_html=True)

        with st.expander("📑 Show source snippets"):
            for src in res["source_documents"]:
                pg = src.metadata.get("page", "?")
                snippet = src.page_content.replace("\n", " ")[:200]
                st.write(f"• **Page {pg}** — _{snippet}…_")

else:
    st.info("📥 Please upload a PDF from the sidebar to begin.")
