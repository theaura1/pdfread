# rag_engine.py
import os
import tempfile
import stat
import oci
import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI

from oci.generative_ai_inference.models import (
    EmbedTextDetails,
    OnDemandServingMode,
    ServingMode,
)

# ── helper to read secrets OR env (works local + Streamlit Cloud) ──────────────
def get_secret(key: str, default: str | None = None):
    return st.secrets[key] if key in st.secrets else os.getenv(key, default)

# ── OCI CONFIG (dict) ─────────────────────────────────────────────────────────
oci_config = {
    "user":        get_secret("OCI_USER"),
    "fingerprint": get_secret("OCI_FINGERPRINT"),
    "key_content": get_secret("OCI_KEY"),
    "tenancy":     get_secret("OCI_TENANCY"),
    "region":      get_secret("OCI_REGION", "us-chicago-1"),
}
TENANCY_ID = oci_config["tenancy"]

# ── Gen-AI inference client (dict works directly) ─────────────────────────────
client = oci.generative_ai_inference.GenerativeAiInferenceClient(oci_config)

service_ep = (
    f"https://inference.generativeai.{oci_config['region']}.oci.oraclecloud.com"
)
client.base_client.endpoint = service_ep

# ── Model IDs ─────────────────────────────────────────────────────────────────
EMBED_ID = get_secret("OCI_EMBEDDINGS_MODEL_ID", "cohere.embed-english-light")
CHAT_ID  = get_secret("OCI_TEXT_MODEL_ID",       "cohere.command-light")

# ── create a temp OCI config file + key so ChatOCIGenAI can auth ──────────────
tmp_dir = tempfile.gettempdir()
pem_path = os.path.join(tmp_dir, "oci_tmp_key.pem")
cfg_path = os.path.join(tmp_dir, "oci_tmp_config")

if not os.path.exists(pem_path):
    with open(pem_path, "w") as f:
        f.write(oci_config["key_content"])
    os.chmod(pem_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600

if not os.path.exists(cfg_path):
    with open(cfg_path, "w") as f:
        f.write(
            "[DEFAULT]\n"
            f"user={oci_config['user']}\n"
            f"fingerprint={oci_config['fingerprint']}\n"
            f"tenancy={oci_config['tenancy']}\n"
            f"region={oci_config['region']}\n"
            f"key_file={pem_path}\n"
        )

# ── CUSTOM EMBEDDINGS USING OCI GEN-AI ────────────────────────────────────────
class OCIEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for chunk in texts:
            details = EmbedTextDetails(
                compartment_id=TENANCY_ID,
                input_type=EmbedTextDetails.INPUT_TYPE_SEARCH_DOCUMENT,
                inputs=[chunk],
                serving_mode=OnDemandServingMode(
                    serving_type=ServingMode.SERVING_TYPE_ON_DEMAND,
                    model_id=EMBED_ID,
                ),
            )
            vectors.append(client.embed_text(details).data.embeddings[0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        details = EmbedTextDetails(
            compartment_id=TENANCY_ID,
            input_type=EmbedTextDetails.INPUT_TYPE_SEARCH_QUERY,
            inputs=[text],
            serving_mode=OnDemandServingMode(
                serving_type=ServingMode.SERVING_TYPE_ON_DEMAND,
                model_id=EMBED_ID,
            ),
        )
        return client.embed_text(details).data.embeddings[0]

# ── DOC LOADING & SPLITTING ───────────────────────────────────────────────────
def load_and_prepare_docs(pdf_path: str):
    pages = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(pages)

# ── RAG CHAIN BUILDER ─────────────────────────────────────────────────────────
def build_qa_chain(docs):
    embeddings  = OCIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    llm = ChatOCIGenAI(
        model_id=CHAT_ID,
        service_endpoint=service_ep,
        compartment_id=TENANCY_ID,
        auth_file_location=cfg_path,  # temp config we just wrote
        auth_profile="DEFAULT",
        model_kwargs={"temperature": 0.0, "max_tokens": 1024},
        is_stream=False,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )
