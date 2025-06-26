# app.py  –– deploy-ready front-end for multi-PDF RAG
import os, textwrap
import streamlit as st

from rag_engine import (
    load_and_prepare_docs_from_multiple_pdfs,
    build_qa_chain,
    cfg_path,        # temp config written by rag_engine.py
    TENANCY_ID,
    CHAT_ID,
    service_ep,
)

from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="📄 Ask Your PDF (OCI GenAI)", page_icon="📄")

# ── THEME (unchanged) ─────────────────────────────────────────────────────────
st.markdown("""<style>/* your long dark-theme CSS exactly as before */</style>""",
            unsafe_allow_html=True)

# ── LLM (cached once) ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOCIGenAI(
        model_id=CHAT_ID,
        service_endpoint=service_ep,
        compartment_id=TENANCY_ID,
        auth_file_location=cfg_path,   # ← temp cfg produced by rag_engine.py
        auth_profile="DEFAULT",
        model_kwargs={"temperature": 0.3, "max_tokens": 768},
        is_stream=False,
    )

LLM = get_llm()

def ask_llm(prompt: str) -> str:
    try:
      return LLM.predict(prompt).strip()
    except Exception as e:
      st.error(f"⚠️ Gen AI error: {e}")
      return ""

# ── Helper transforms ─────────────────────────────────────────────────────────
def translate(txt, lang):
    if lang in ("None", "English"): return txt
    return ask_llm(f"Translate into {lang}:\n\n{txt}")

def summarise(docs):
    corpus = " ".join(d.page_content for d in docs)[:1000]
    return ask_llm("Give a ~200-word summary:\n\n" + corpus)

def easify(txt):
    short = txt[:1000]
    return ask_llm("Explain like I'm 5 in simple words:\n\n" + short)


# ── SIDEBAR CONTROLS ──────────────────────────────────────────────────────────
st.sidebar.header("🔧 Controls")
files = st.sidebar.file_uploader("Upload PDFs", ["pdf"], accept_multiple_files=True)
lang  = st.sidebar.selectbox("Translate output to:", 
                             ["None", "English", "Hindi", "Tamil", "French"], 0)
st.sidebar.markdown("---")
st.sidebar.write("Built with OCI GenAI · Lavkesh")

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "txt" not in st.session_state:
    st.session_state.txt   = None
    st.session_state.title = None
    st.session_state.snips = None   # source snippets for Q-A only

# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
st.title("📄 Ask Your PDF")

if not files:
    st.info("📥 Upload one or more PDFs from the sidebar to begin.")
    st.stop()

# save uploads to ./data
os.makedirs("data", exist_ok=True)
paths = []
for uf in files:
    p = os.path.join("data", uf.name)
    with open(p, "wb") as o:
        o.write(uf.getvalue())
    paths.append(p)

# Build documents & RAG chain (no caching for docs/vectorstore)
docs = load_and_prepare_docs_from_multiple_pdfs(paths)
qa   = build_qa_chain(docs)

# ── ACTION BUTTONS ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
if col1.button("📝 Summarise PDFs"):
    with st.spinner("Summarising…"):
        st.session_state.txt   = summarise(docs)
        st.session_state.title = "Summary"
        st.session_state.snips = None

question = st.text_input("💬 Ask a question:")
if question:
    with st.spinner("Answering…"):
        res = qa(question)
    st.session_state.txt   = res["result"]
    st.session_state.title = "Answer"
    st.session_state.snips = res["source_documents"]

# ── DISPLAY OUTPUT ────────────────────────────────────────────────────────────
if st.session_state.txt:
    st.subheader(st.session_state.title)
    st.markdown(f"<div class='answer-box'>{st.session_state.txt}</div>", 
                unsafe_allow_html=True)

    if st.button("🧸 Easify this"):
        with st.spinner("Simplifying…"):
            st.session_state.txt   = easify(st.session_state.txt)
            st.session_state.title = "Explained Easily"
            st.session_state.snips = None
        st.subheader(st.session_state.title)
        st.markdown(f"<div class='answer-box'>{st.session_state.txt}</div>",
                    unsafe_allow_html=True)

    # translation
    trans = translate(st.session_state.txt, lang)
    if trans != st.session_state.txt:
        st.subheader(f"Translated ({lang})")
        st.markdown(f"<div class='answer-box'>{trans}</div>", 
                    unsafe_allow_html=True)

    # snippets only for Q-A answers
    if st.session_state.snips:
        with st.expander("📑 Source snippets"):
            for s in st.session_state.snips:
                pg   = s.metadata.get("page", "?")
                name = s.metadata.get("source", "doc")
                snip = textwrap.shorten(s.page_content.replace("\n", " "), 150)
                st.write(f"• **{name} – page {pg}** — _{snip}_")
