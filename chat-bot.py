"""Streamlit RAG chatbot over Word and PDF documents in ./docs.

Run with: streamlit run rag_chatbot.py
Requires GROQ_API_KEY in the environment and the project's LangChain/Chroma,
sentence-transformers, and Streamlit dependencies.
"""

import hashlib
import os
import re
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import chromadb
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import base64
from dotenv import dotenv_values, load_dotenv

def get_api_key():
    """Retrieve Groq API key securely from multiple fallback sources."""
    # 1. Check local .env file
    env_file_key = dotenv_values(".env").get("GROQ_API_KEY")
    if env_file_key:
        return env_file_key

    # 2. Check Streamlit cloud secrets
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    # 3. Check OS environment variables
    return os.getenv("GROQ_API_KEY")


GROQ_API_KEY = get_api_key()
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION = "udx_rag_1000_150"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_TOKENS = 1000
OVERLAP_TOKENS = 100
TOP_K = 4
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
ROLE_INSTRUCTIONS = {
    "Technical Product Owner": (
        "Respond as a Technical Product Owner. Emphasize requirements, user "
        "impact, acceptance criteria, scope, and tradeoffs. Keep the response concise and to the point."
        "Do not provide respone for 401k related questions."
    ),
    "Software Developer": (
        "Respond as a Lead software developer. Emphasize implementation details, "
        "interfaces, edge cases, and technical constraints. Keep the response concise and to the point."
        "Do not provide respone for 401k related questions."
    ),
    "QA Analyst": (
        "Respond as a Lead QA Analyst. Emphasize on Test case creation, best practices, happy path cases, "
        "edge cases, and Automation testing. Keep the response concise and to the point."
        "Do not provide respone for 401k related questions."
    ),
    "401K Assitant": (
        "Respond as a 401K Assitant. Provide information about Capital Accumulation Plan and 401K Regulations."
        "Do not respond about any other projects or coding questions. Keep the response concise and to the point."
    ),
}


def read_docx(path: Path) -> str:
    """Extract readable paragraphs from a .docx file using its XML."""
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as docx:
        root = ET.fromstring(docx.read("word/document.xml"))
    paragraphs = []
    for paragraph in root.findall(".//w:p", ns):
        value = "".join(node.text or "" for node in paragraph.findall(".//w:t", ns))
        value = re.sub(r"\s+", " ", value).strip()
        if value:
            paragraphs.append(value)
    return "\n".join(paragraphs)


def read_pdf(path: Path) -> str:
    """Extract searchable text from a PDF (scanned PDFs need OCR first)."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def read_document(path: Path) -> str:
    """Read a supported source file from docs/."""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return read_docx(path)
    if suffix == ".pdf":
        return read_pdf(path)
    raise ValueError(f"Unsupported document type: {path.suffix}")


def chunk_text(text: str, tokenizer) -> list[str]:
    """Split text into 1000-token chunks with 80-token overlap."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    for start in range(0, len(token_ids), step):
        piece = token_ids[start : start + CHUNK_TOKENS]
        if not piece:
            break
        chunks.append(tokenizer.decode(piece, skip_special_tokens=True))
        if start + CHUNK_TOKENS >= len(token_ids):
            break
    return chunks


@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_tokenizer():
    # Use the same tokenizer as the local embedding model for deterministic
    # chunk sizing without requiring an additional tokenizer package.
    return get_embedder().tokenizer


def ingest_documents():
    """Index every DOCX and PDF in docs/, refreshing changed source files."""
    if not DOCS_DIR.is_dir():
        raise FileNotFoundError(f"Docs folder not found: {DOCS_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        COLLECTION, metadata={"description": "RAG chunks from docs/*.docx"}
    )
    embedder = get_embedder()
    tokenizer = get_tokenizer()
    paths = sorted(
        path
        for path in DOCS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".docx", ".pdf"}
    )
    if not paths:
        raise FileNotFoundError(f"No .docx or .pdf documents found in {DOCS_DIR}")

    for path in paths:
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        prior = collection.get(where={"source": path.name}, include=["metadatas"])
        ids = prior["ids"]
        metadata = prior.get("metadatas") or []
        if ids and all(item.get("file_hash") == file_hash for item in metadata):
            continue
        if ids:
            collection.delete(ids=ids)
        extracted_text = read_document(path)
        if not extracted_text:
            st.warning(f"No extractable text in {path.name}; it may be a scanned PDF.")
            continue
        chunks = chunk_text(extracted_text, tokenizer)
        if not chunks:
            continue
        vectors = embedder.encode(chunks, show_progress_bar=False).tolist()
        collection.add(
            ids=[f"{path.stem}-{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=vectors,
            metadatas=[
                {"source": path.name, "chunk": i, "file_hash": file_hash}
                for i in range(len(chunks))
            ],
        )
    return collection


def retrieve(collection, question: str):
    vector = get_embedder().encode([question]).tolist()[0]
    result = collection.query(
        query_embeddings=[vector],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    return list(zip(result["documents"][0], result["metadatas"][0]))


def answer_question(client, collection, question: str, role: str, history: list[dict]):
    matches = retrieve(collection, question)
    context = "\n\n".join(
        f"[Source: {meta['source']}, chunk {meta['chunk']}]\n{text}"
        for text, meta in matches
    )
    messages = [
        {
            "role": "system",
            "content": (
                f"{ROLE_INSTRUCTIONS[role]} Answer using the supplied document "
                "context and relevant conversation history. If the documents "
                "do not support an answer, say so. Do not invent citations. "
                f"\n\nRetrieved document context:\n{context}"
            ),
        }
    ]
    # The UI retains and supplies at most the last 10 messages (five turns).
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.2, max_tokens=700
    )
    return (
        response.choices[0].message.content
        or "I couldn't generate a response. Pick the appropriate role for your question."
    )


def get_data_uri(file_path):
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


bg_image = get_data_uri("resources/BG.jpg")

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url('{bg_image}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stAppDeployButton, .stDeployButton {{
        visibility: hidden;
    }}
    [data-testid="stToolbar"] {{
        visibility: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    st.set_page_config(page_title="UDX RAG Chatbot", page_icon="🤖")
    st.title("UDX Project Assistant")
    st.caption(
        "All answers are grounded in the reference documents provided within the directory"
    )
    role = st.selectbox("User role", list(ROLE_INSTRUCTIONS))

    api_key = GROQ_API_KEY
    if not api_key:
        st.error("Set the GROQ_API_KEY environment variable to enable chat.")
        st.stop()

    try:
        collection = ingest_documents()
    except Exception as exc:
        st.error(f"Could not prepare the document index: {exc}")
        st.stop()

    st.caption(
        f"Indexed {collection.count()} chunks · {CHUNK_TOKENS} tokens each · {OVERLAP_TOKENS} token overlap · Top {TOP_K} chunks set as Context"
    )
    # with st.expander("Inspect indexed chunk metadata"):
    #     indexed = collection.get(
    #         include=["documents", "metadatas"],
    #     )
    #     for chunk_id, metadata, document in zip(
    #         indexed["ids"], indexed["metadatas"], indexed["documents"]
    #     ):
    #         st.markdown(f"**{chunk_id}**")
    #         st.json(metadata)
    #         with st.expander(f"Chunk text · {chunk_id}"):
    #             st.write(document)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask questions about your UDX Project."):
        history = st.session_state.messages[-5:]
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching the documents..."):
                try:
                    client = Groq(api_key=api_key)
                    response = answer_question(
                        client, collection, prompt, role, history
                    )
                except Exception as exc:
                    response = f"I couldn't complete the request: {exc}"
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Bound memory as well as the context sent to the model.
        st.session_state.messages = st.session_state.messages[-10:]

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
