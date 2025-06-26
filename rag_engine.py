# rag_engine.py  ── Streamlit-Cloud-ready multi-PDF RAG engine
import os, tempfile, stat
import streamlit as st
import oci

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI

from oci.generative_ai_inference.models import (
    EmbedTextDetails,
    OnDemandServingMode,
    ServingMode,
)

# ── helper: fetch from st.secrets ➊ else env ➋ else default ➌ ────────────────
def _secret(key: str, default: str | None = None):
    return st.secrets[key] if key in st.secrets else os.getenv(key, default)

# ── OCI BASIC CONFIG (dict style) ─────────────────────────────────────────────
oci_cfg = {
    "user":        _secret("OCI_USER"),
    "fingerprint": _secret("OCI_FINGERPRINT"),
    "key_content": _secret("OCI_KEY"),
    "tenancy":     _secret("OCI_TENANCY"),
    "region":      _secret("OCI_REGION", "us-chicago-1"),
}
TENANCY_ID  = oci_cfg["tenancy"]
EMBED_ID    = _secret("OCI_EMBEDDINGS_MODEL_ID", "cohere.embed-english-light")
CHAT_ID     = _secret("OCI_TEXT_MODEL_ID",       "cohere.command-light")

# ── Low-level inference client (for embeddings) ───────────────────────────────
client = oci.generative_ai_inference.GenerativeAiInferenceClient(oci_cfg)
service_ep = f"https://inference.generativeai.{oci_cfg['region']}.oci.oraclecloud.com"
client.base_client.endpoint = service_ep

# ── temp key & config so ChatOCIGenAI (langchain wrapper) can auth ────────────
tmp_dir   = tempfile.gettempdir()
pem_path  = os.path.join(tmp_dir, "oci_tmp_key.pem")
cfg_path  = os.path.join(tmp_dir, "oci_tmp_config")

if not os.path.exists(pem_path):
    with open(pem_path, "w") as f:
        f.write(oci_cfg["key_content"])
    os.chmod(pem_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600

if not os.path.exists(cfg_path):
    with open(cfg_path, "w") as f:
        f.write(
            "[DEFAULT]\n"
            f"user={oci_cfg['user']}\n"
            f"fingerprint={oci_cfg['fingerprint']}\n"
            f"tenancy={oci_cfg['tenancy']}\n"
            f"region={oci_cfg['region']}\n"
            f"key_file={pem_path}\n"
        )

# ── Embedding adapter using OCI GenAI embeddings endpoint ─────────────────────
class OCIEmbeddings(Embeddings):
    def embed_documents(self, texts):
        vecs = []
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
            vecs.append(client.embed_text(details).data.embeddings[0])
        return vecs

    def embed_query(self, text):
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

# ── PDF -> page chunks (metadata keeps page & file name) ──────────────────────
def _chunk_page(text: str, page_no: int, src: str,
                chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    idx = 0
    for chunk in splitter.split_text(text):
        start = text.find(chunk, idx)
        end   = start + len(chunk)
        idx   = start + 1
        yield Document(
            page_content=chunk,
            metadata={
                "page":       page_no,
                "char_start": start,
                "char_end":   end,
                "source":     src,
            },
        )

def load_and_prepare_docs_from_multiple_pdfs(paths: list[str]):
    docs: list[Document] = []
    for p in paths:
        pages   = PyPDFLoader(p).load()
        srcname = os.path.basename(p)
        for pg in pages:
            page_n = pg.metadata.get("page", 0)
            docs.extend(_chunk_page(pg.page_content, page_n, srcname))
    return docs

# ── Build the Retrieval-Augmented QA chain ────────────────────────────────────
def build_qa_chain(docs):
    vec_store = FAISS.from_documents(docs, OCIEmbeddings())

    llm = ChatOCIGenAI(
        model_id=CHAT_ID,
        service_endpoint=service_ep,
        compartment_id=TENANCY_ID,
        auth_file_location=cfg_path,  # temp config we wrote
        auth_profile="DEFAULT",
        model_kwargs={"temperature": 0.0, "max_tokens": 1024},
        is_stream=False,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vec_store.as_retriever(),
        return_source_documents=True,
    )
