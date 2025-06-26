# app.py  –– PDF Assistant + Chat + geo clock  (OCI Gen-AI back-end)
import os, textwrap, requests, datetime, pytz
import streamlit as st
import streamlit.components.v1 as components
from rag_engine import (
    load_and_prepare_docs_from_multiple_pdfs,
    build_qa_chain,
    cfg_path, TENANCY_ID, CHAT_ID, service_ep,
)
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI

# ────────────────────────────────────────────────────────────────────
# PAGE CONFIG & DARK THEME CSS (unchanged)
# ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ask Your PDF (OCI GenAI)", page_icon="✨")

DARK_CSS = r"""<style> /* (same CSS you already have) */ </style>"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────
# LLM (cached once)
# ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOCIGenAI(
        model_id=CHAT_ID,
        service_endpoint=service_ep,
        compartment_id=TENANCY_ID,
        auth_file_location=cfg_path,
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

# ────────────────────────────────────────────────────────────────────
# Helper transforms
# ────────────────────────────────────────────────────────────────────
def translate(txt, lang):
    if lang in ("None", "English"):
        return txt
    return ask_llm(f"Translate into {lang}:\n\n{txt}")

def summarise(docs):
    corpus = " ".join(d.page_content for d in docs)[:1000]
    return ask_llm("Give a ~200-word summary:\n\n" + corpus)

def easify(txt):
    short = txt[:1000]
    return ask_llm("Explain like I'm 5 in simple words:\n\n" + short)

# ────────────────────────────────────────────────────────────────────
# IP / GEO helper
# ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_ip_geo():
    try:
        data = requests.get("https://ipapi.co/json/", timeout=3).json()
        return {
            "ip": data.get("ip", "–"),
            "city": data.get("city", ""),
            "country": data.get("country_name", ""),
            "utc_offset": data.get("utc_offset", "+00:00"),
        }
    except Exception:
        return {"ip": "Unavailable", "city": "", "country": "", "utc_offset": "+00:00"}

# ────────────────────────────────────────────────────────────────────
# SIDEBAR: geo block + live clock + mode selector
# ────────────────────────────────────────────────────────────────────
geo = get_ip_geo()

with st.sidebar:
    components.html(
        f"""
        <div style="font-family:Inter,sans-serif;font-weight:300;color:#e0e0e0;">
          <div style="margin-bottom:4px">
              🌐 Your IP: <span style="color:#12c26a;">{geo['ip']}</span>
          </div>
          <div style="margin-bottom:4px">
              📍 Location: {geo['city']} {geo['country']}
          </div>
          <div style="margin-bottom:0">
              🕒 Current time: <span id='liveclock'></span>
          </div>
        </div>
        <script>
          function tick(){{
            const now=new Date();
            const opts={{hour:'2-digit',minute:'2-digit',
                         second:'2-digit',hour12:true}};
            document.getElementById('liveclock').textContent=
              now.toLocaleTimeString([],opts);
          }}
          tick(); setInterval(tick,1000);
        </script>
        """,
        height=90,
    )

mode = st.sidebar.radio(
    "Choose mode",
    ["📄 PDF Assistant", "🤖 Chat with AI"],
    horizontal=True,
    index=0,
)
st.sidebar.markdown("---")

# ────────────────────────────────────────────────────────────────────
# SESSION STATE for PDF & Chat
# ────────────────────────────────────────────────────────────────────
if "txt"   not in st.session_state: st.session_state.txt   = None
if "title" not in st.session_state: st.session_state.title = None
if "snips" not in st.session_state: st.session_state.snips = None
if "chat"  not in st.session_state: st.session_state.chat  = []

# ────────────────────────────────────────────────────────────────────
# MAIN TITLE
# ────────────────────────────────────────────────────────────────────
st.title("📄 Ask Your PDF" if mode.startswith("📄") else "🤖 Chat with AI")

# --------------------------------------------------------------------
# CHAT WITH AI MODE
# --------------------------------------------------------------------
if mode.startswith("🤖"):

    # display history
    for who, msg in st.session_state.chat:
        st.markdown(f"**{who}:** {msg}")

    col_msg, col_btn = st.columns([4, 1])
    user_msg = col_msg.text_input("Your message", key="chat_input")
    if col_btn.button("Send", key="send_btn") and user_msg.strip():
        with st.spinner("AI is typing…"):
            reply = ask_llm(user_msg.strip())
        st.session_state.chat.extend(
            [("You", user_msg.strip()), ("AI", reply)]
        )
        # clear input then rerun
        del st.session_state["chat_input"]
        st.experimental_rerun()

    st.stop()   # do not execute PDF logic below

# --------------------------------------------------------------------
# PDF ASSISTANT MODE
# --------------------------------------------------------------------
if mode.startswith("📄"):

    # sidebar controls for PDF mode
    st.sidebar.header("Controls")
    files = st.sidebar.file_uploader(
        "Upload PDFs", ["pdf"], accept_multiple_files=True, key="pdf_files")
    lang = st.sidebar.selectbox(
        "Translate to",
        ["None", "English", "Hindi", "Tamil", "French"],
        0,
        key="lang_select",
    )

    if not files:
        st.info("📥 Upload one or more PDFs from the sidebar to begin.")
        st.stop()

    # save uploads to ./data
    os.makedirs("data", exist_ok=True)
    paths = []
    for uf in files:
        path = os.path.join("data", uf.name)
        with open(path, "wb") as f:
            f.write(uf.getvalue())
        paths.append(path)

    docs = load_and_prepare_docs_from_multiple_pdfs(paths)
    qa   = build_qa_chain(docs)

    # ACTION BUTTONS
    col1, _ = st.columns(2)
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

    # DISPLAY OUTPUT
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

        trans = translate(st.session_state.txt, lang)
        if trans != st.session_state.txt:
            st.subheader(f"Translated ({lang})")
            st.markdown(f"<div class='answer-box'>{trans}</div>",
                        unsafe_allow_html=True)

        if st.session_state.snips:
            with st.expander("📑 Source snippets"):
                for s in st.session_state.snips:
                    pg   = s.metadata.get("page", "?")
                    name = s.metadata.get("source", "doc")
                    snip = textwrap.shorten(
                        s.page_content.replace("\n", " "), 150)
                    st.write(f"• **{name} – page {pg}** — _{snip}_")
