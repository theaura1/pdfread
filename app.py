# app.py  –– PDF Assistant + Chat + geo clock  (OCI Gen-AI back-end)
import os, textwrap, requests
import streamlit as st
import streamlit.components.v1 as components
from rag_engine import (
    load_and_prepare_docs_from_multiple_pdfs,
    build_qa_chain,
    cfg_path, TENANCY_ID, CHAT_ID, service_ep,
)
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  +  DARK-MODE THEME (identical to your local file)
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ask Your PDF (OCI GenAI)", page_icon="✨")

DARK_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

/* Moving gradient background */
[data-testid="stAppViewContainer"] {
  font-family: 'Inter', sans-serif;
  background: linear-gradient(-45deg,#1a1a2e,#0f0f1e,#1f1f39,#0f0f1e);
  background-size: 400% 400%;
  animation: gradientBG 20s ease infinite;
  color:#e0e0e0;
}
@keyframes gradientBG{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}

/* Sidebar */
[data-testid="stSidebar"]{background:rgba(26,26,46,.95);border-right:1px solid #333}

/* Main width */
.block-container{max-width:850px;padding:2rem 1rem}

/* ONLY target the Send button inside .send-btn-wrapper */
.send-btn-wrapper button {
    white-space: nowrap;
    padding: 0.45rem 1.2rem !important;
    margin-left: auto;
    margin-right: auto;
    margin-top: 10px;
    min-width: 90px;           /* wider but not full width */
    text-align: center;
}




/* Title */
.stMarkdown h1{
  font-size:3rem;font-weight:800;text-align:center;margin-bottom:.3rem;
  background:linear-gradient(90deg,#ff6ec4,#7373ff);-webkit-background-clip:text;color:transparent;
}

/* Answer box */
.answer-box{
  position:relative;background:rgba(42,42,63,.8);border:4px solid transparent;border-radius:12px;
  padding:1.5rem;box-shadow:0 4px 30px rgba(0,0,0,.5);
  animation:fadeIn .6s ease-out,rotateBorder 8s linear infinite;--bdeg:0deg;
  border-image:linear-gradient(var(--bdeg),#ff6ec4,#7373ff,#18dcff)1;color:#f5f5f5;margin-bottom:1rem;
}
@keyframes fadeIn{from{opacity:0;transform:translateY(-10px)}to{opacity:1;transform:translateY(0)}}
@keyframes rotateBorder{to{--bdeg:360deg}}

/* Inputs */
.stTextInput > label{color:#ddd!important}
.stTextInput > div > input{
  background:rgba(42,42,63,.7)!important;border:1px solid #555!important;border-radius:8px!important;
  color:#e0e0e0!important;padding:.5rem 1rem!important;
}

/* Buttons */
.stButton > button{
  background:linear-gradient(90deg,#ff6ec4,#7373ff)!important;border:none!important;border-radius:8px!important;
  padding:.6rem 1.2rem!important;font-weight:600!important;
  box-shadow:0 4px 15px rgba(0,0,0,.4)!important;transition:transform .2s,box-shadow .2s!important;
}
.stButton > button:hover{transform:translateY(-2px)!important;box-shadow:0 6px 20px rgba(0,0,0,.6)!important}

/* ░░░ your existing dark-theme CSS (unchanged) ░░░ */

/* ───── NEW: breathing-room tweaks ───── */

/* add gap after the IP/geo block */
[data-testid="stSidebar"] > div:first-child {
  margin-bottom: 1rem;           /* ≈16 px – tweak as you like */
}

/* space above and below the whole radio selector */
.stRadio {
  margin-top: 0.75rem !important;
  margin-bottom: 0.75rem !important;
}

/* optional: loosen the two radio choices a bit */
.stRadio > div {
  row-gap: .35rem !important;    /* vertical gap between options */
}

/* give the custom-HTML iframe some breathing room */
[data-testid="stSidebar"] iframe {
    margin-bottom: 1rem !important;   /* adjust to taste */
}

/* (optional) extra padding for the radio block itself */
.stRadio {
    margin-top: 0.5rem !important;
}

/* Expander */
.stExpanderHeader{
  background:rgba(42,42,63,.7)!important;border:1px solid #444!important;border-radius:8px!important;
  padding:.5rem 1rem!important;color:#ddd!important;
}
.stExpanderHeader:hover{background:rgba(42,42,63,.9)!important}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LLM (cached)
# ─────────────────────────────────────────────────────────────
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
        st.error(f"⚠️ Gen-AI error: {e}")
        return ""

# ─────────────────────────────────────────────────────────────
# Helper transforms
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# IP / GEO helper
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# SIDEBAR  (geo + clock + mode)
# ─────────────────────────────────────────────────────────────
geo = get_ip_geo()
with st.sidebar:
    # 1) bump the iframe height from 90 → 110 px (adds ~20 px blank space)
    components.html(
        f"""
        <div style="font-family:Inter,sans-serif;font-weight:50;color:#e0e0e0;">
          <div style="margin-bottom:4px">🌐 Your IP: <span style="color:#12c26a;">{geo['ip']}</span></div>
          <div style="margin-bottom:4px">📍 Location: {geo['city']} {geo['country']}</div>
          <div style="margin-bottom:4px">🕒 Current time: <span id='liveclock'></span></div>
        </div>
        <script>
          function tick(){{
            const now=new Date();
            const opts={{hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:true}};
            document.getElementById('liveclock').textContent=now.toLocaleTimeString([],opts);
          }}
          tick(); setInterval(tick,1000);
        </script>
        """,
        height=110,          # <── increased from 90
    )

    # 2) optional extra spacer (≈ 12–14 px)
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

# rest of the sidebar widgets
mode = st.sidebar.radio(
    "Choose mode",
    ["📄 PDF Assistant", "🤖 Chat with AI"],
    horizontal=True,
    index=0,
)
st.sidebar.markdown("---")


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "txt"   not in st.session_state: st.session_state.txt   = None
if "title" not in st.session_state: st.session_state.title = None
if "snips" not in st.session_state: st.session_state.snips = None
if "chat"  not in st.session_state: st.session_state.chat  = []

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
st.title("📄 Ask Your PDF" if mode.startswith("📄") else "🤖 Chat with AI")

# -----------------------------------------------------------------
# ── CHAT WITH AI MODE ───────────────────────────────────────────
# ── CHAT WITH AI MODE ───────────────────────────────────────────
if mode.startswith("🤖"):

    if "chat" not in st.session_state:
        st.session_state.chat = []

    # show chat history
    for who, msg in st.session_state.chat:
        st.markdown(f"**{who}:** {msg}")

    # ✔️ Better layout: 90% + 10% split with padding
    col_msg, col_btn = st.columns((10, 1), gap="small")

    with col_msg:
        user_msg = st.text_input("Your message", key="chat_input")

    with col_btn:
        st.markdown("""
            <div class="send-btn-wrapper" style="padding-top: 10px;">
        """, unsafe_allow_html=True)
    
        send_clicked = st.button("Send", key="send_btn")
    
        st.markdown("</div>", unsafe_allow_html=True)

    if send_clicked and user_msg.strip():
        with st.spinner("AI is typing…"):
            reply = ask_llm(user_msg.strip())
        st.session_state.chat.extend([("You", user_msg.strip()), ("AI", reply)])

        del st.session_state["chat_input"]
        (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()

    st.stop()



# -----------------------------------------------------------------
# PDF ASSISTANT MODE
# -----------------------------------------------------------------
if mode.startswith("📄"):

    st.sidebar.header("Controls")
    files = st.sidebar.file_uploader(
        "Upload PDFs", ["pdf"], accept_multiple_files=True, key="pdf_files")
    lang = st.sidebar.selectbox(
        "Translate to", ["None", "English", "Hindi", "Tamil", "French"], 0,
        key="lang_select",
    )

    if not files:
        st.info("📥 Upload one or more PDFs from the sidebar to begin.")
        st.stop()

    os.makedirs("data", exist_ok=True)
    paths = []
    for uf in files:
        p = os.path.join("data", uf.name)
        with open(p, "wb") as f:
            f.write(uf.getvalue())
        paths.append(p)

    docs = load_and_prepare_docs_from_multiple_pdfs(paths)
    qa   = build_qa_chain(docs)

    # Action buttons
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

    # Display
    if st.session_state.txt:
        st.subheader(st.session_state.title)
        st.markdown(f"<div class='answer-box'>{st.session_state.txt}</div>", unsafe_allow_html=True)

        if st.button("🧸 Easify this"):
            with st.spinner("Simplifying…"):
                st.session_state.txt   = easify(st.session_state.txt)
                st.session_state.title = "Explained Easily"
                st.session_state.snips = None
            st.subheader(st.session_state.title)
            st.markdown(f"<div class='answer-box'>{st.session_state.txt}</div>", unsafe_allow_html=True)

        trans = translate(st.session_state.txt, lang)
        if trans != st.session_state.txt:
            st.subheader(f"Translated ({lang})")
            st.markdown(f"<div class='answer-box'>{trans}</div>", unsafe_allow_html=True)

        if st.session_state.snips:
            with st.expander("📑 Source snippets"):
                for s in st.session_state.snips:
                    pg   = s.metadata.get("page", "?")
                    name = s.metadata.get("source", "doc")
                    snip = textwrap.shorten(s.page_content.replace("\n", " "), 150)
                    st.write(f"• **{name} – page {pg}** — _{snip}_")
