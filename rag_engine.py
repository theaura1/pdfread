# rag_engine.py

import os
import oci
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI

from oci.generative_ai_inference.models import (
    EmbedTextDetails,
    OnDemandServingMode,
    ServingMode,
)

# ── LOAD ENV & OCI CONFIG ─────────────────────────────────────────────────────
load_dotenv()

config = oci.config.from_file(
    os.getenv("OCI_CONFIG_FILE", "./oci/config"),
    profile_name=os.getenv("OCI_PROFILE", "DEFAULT"),
)
client = oci.generative_ai_inference.GenerativeAiInferenceClient(config)

# ── FORCE THE INFERENCE ENDPOINT ───────────────────────────────────────────────
service_ep = os.getenv(
    "OCI_SERVICE_ENDPOINT",
    "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
)
client.base_client.endpoint = service_ep

# ── MODEL IDS ──────────────────────────────────────────────────────────────────
EMBED_ID = os.getenv("OCI_EMBEDDINGS_MODEL_ID")      # e.g. cohere.embed-english-light or its OCID
CHAT_ID  = os.getenv("OCI_TEXT_MODEL_ID")            # e.g. cohere.command-light or its OCID

# ── CUSTOM EMBEDDINGS USING OCI GEN AI ────────────────────────────────────────
class OCIEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for chunk in texts:
            details = EmbedTextDetails(
                compartment_id=config["tenancy"],
                input_type=EmbedTextDetails.INPUT_TYPE_SEARCH_DOCUMENT,
                inputs=[chunk],
                serving_mode=OnDemandServingMode(
                    serving_type=ServingMode.SERVING_TYPE_ON_DEMAND,
                    model_id=EMBED_ID,
                ),
            )
            resp = client.embed_text(details)
            vectors.append(resp.data.embeddings[0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        details = EmbedTextDetails(
            compartment_id=config["tenancy"],
            input_type=EmbedTextDetails.INPUT_TYPE_SEARCH_QUERY,
            inputs=[text],
            serving_mode=OnDemandServingMode(
                serving_type=ServingMode.SERVING_TYPE_ON_DEMAND,
                model_id=EMBED_ID,
            ),
        )
        resp = client.embed_text(details)
        return resp.data.embeddings[0]


# ── DOCUMENT LOADING & SPLITTING ───────────────────────────────────────────────
def load_and_prepare_docs(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(pages)


# ── BUILD THE RAG CHAIN ────────────────────────────────────────────────────────
def build_qa_chain(docs):
    embeddings  = OCIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    llm = ChatOCIGenAI(
        model_id=CHAT_ID,
        service_endpoint=service_ep,
        compartment_id=config["tenancy"],
        auth_file_location=os.getenv("OCI_CONFIG_FILE", "./oci/config"),
        auth_profile      =os.getenv("OCI_PROFILE",      "DEFAULT"),
        model_kwargs={"temperature": 0.0, "max_tokens": 1024},
        is_stream=False,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )
