import os, textwrap
import streamlit as st
from dotenv import load_dotenv
import oci

from rag_engine import load_and_prepare_docs_from_multiple_pdfs, build_qa_chain
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ask Your PDF (OCI GenAI)", page_icon="✨")

# (your CSS unchanged) ---------------------------------------------------------
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

# ── OCI CONFIG ────────────────────────────────────────────────────────────────
load_dotenv()
CFG_FILE   = os.getenv("OCI_CONFIG_FILE", "./oci/config")
CFG_PROF   = os.getenv("OCI_PROFILE", "DEFAULT")
SERVICE_EP = os.getenv("OCI_SERVICE_ENDPOINT")
CHAT_ID    = os.getenv("OCI_TEXT_MODEL_ID")
TENANCY_ID = oci.config.from_file(CFG_FILE, profile_name=CFG_PROF)["tenancy"]

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOCIGenAI(
        model_id=CHAT_ID,
        service_endpoint=SERVICE_EP,
        compartment_id=TENANCY_ID,
        auth_file_location=CFG_FILE,
        auth_profile=CFG_PROF,
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

def translate(text, lang):
    if lang in ("None", "English"): return text
    return ask_llm(f"Translate into {lang}:\n\n{text}")

def summarise(docs):
    corpus = " ".join(d.page_content for d in docs)[:1000]
    return ask_llm("Give a ~200-word summary:\n\n" + corpus)

def easify(text):
    short = text[:1000]
    return ask_llm("Explain like I'm 5 in simple words:\n\n" + short)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.header("Controls")
files = st.sidebar.file_uploader("Upload PDFs", ["pdf"], accept_multiple_files=True)
lang  = st.sidebar.selectbox("Translate to", ["None","English","Hindi","Tamil","French"], 0)

# ── SESSION -------------------------------------------------------------------
if "txt" not in st.session_state:
    st.session_state.txt    = None
    st.session_state.title  = None
    st.session_state.snips  = None   # store snippets only for Q-A

# ── MAIN ----------------------------------------------------------------------
st.title("📄 Ask Your PDF")
if not files:
    st.info("Upload PDFs first.")
    st.stop()

paths = []
os.makedirs("data", exist_ok=True)
for f in files:
    p = os.path.join("data", f.name)
    with open(p,"wb") as o: o.write(f.getvalue())
    paths.append(p)

docs = load_and_prepare_docs_from_multiple_pdfs(paths)
qa   = build_qa_chain(docs)

# action buttons
col1, col2 = st.columns(2)
if col1.button("📝 Summarise PDFs"):
    with st.spinner("Summarising…"):
        st.session_state.txt   = summarise(docs)
        st.session_state.title = "Summary"
        st.session_state.snips = None

question = st.text_input("Ask a question:")
if question:
    with st.spinner("Answering…"):
        res = qa(question)
    st.session_state.txt   = res["result"]
    st.session_state.title = "Answer"
    st.session_state.snips = res["source_documents"]

# show current text
if st.session_state.txt:
    st.subheader(st.session_state.title)
    st.markdown(f"<div class='answer-box'>{st.session_state.txt}</div>", unsafe_allow_html=True)

    if st.button("🧸 Easify this"):
        with st.spinner("Simplifying…"):
            st.session_state.txt = easify(st.session_state.txt)
            st.session_state.title = "Explained Easily"
            st.session_state.snips = None
        # Immediately display the new simpler text
        st.subheader(st.session_state.title)
        st.markdown(f"<div class='answer-box'>{st.session_state.txt}</div>", unsafe_allow_html=True)

    # translation (always based on current displayed text)
    trans = translate(st.session_state.txt, lang)
    if trans != st.session_state.txt:
        st.subheader(f"Translated ({lang})")
        st.markdown(f"<div class='answer-box'>{trans}</div>", unsafe_allow_html=True)

    # source snippets only for Q-A
    if st.session_state.snips:
        with st.expander("Source snippets"):
            for s in st.session_state.snips:
                pg   = s.metadata.get("page","?")
                name = s.metadata.get("source","doc")
                snip = textwrap.shorten(s.page_content.replace("\n"," "), 150)
                st.write(f"• **{name} – page {pg}** — _{snip}_")
