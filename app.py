from __future__ import annotations

import hashlib
from io import BytesIO

import streamlit as st
from docx import Document as WordDocument
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def extract_text(file_name: str, file_bytes: bytes) -> list[Document]:
    """Extract text from a supported upload and preserve page metadata when available."""
    extension = file_name.rsplit(".", maxsplit=1)[-1].lower()

    if extension == "pdf":
        pages = PdfReader(BytesIO(file_bytes)).pages
        documents = [
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": file_name, "page": page_number},
            )
            for page_number, page in enumerate(pages, start=1)
        ]
    elif extension == "docx":
        word_document = WordDocument(BytesIO(file_bytes))
        paragraphs = [paragraph.text for paragraph in word_document.paragraphs]
        tables = [
            " | ".join(cell.text for cell in row.cells)
            for table in word_document.tables
            for row in table.rows
        ]
        text = "\n".join(part for part in paragraphs + tables if part.strip())
        documents = [Document(page_content=text, metadata={"source": file_name})]
    else:
        text = file_bytes.decode("utf-8-sig")
        documents = [Document(page_content=text, metadata={"source": file_name})]

    if not any(document.page_content.strip() for document in documents):
        raise ValueError("No readable text was found in the uploaded document.")

    return documents


def format_context(documents: list[Document]) -> str:
    """Format retrieved chunks with their source locations for answer citations."""
    excerpts = []
    for document in documents:
        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page")
        location = f"{source}, page {page}" if page is not None else source
        excerpts.append(f"[Source: {location}]\n{document.page_content}")
    return "\n\n---\n\n".join(excerpts)


st.set_page_config(page_title="Document Q&A")
st.title("Ask your document")
st.write("Upload a PDF, Word document, or text file, then ask questions about it.")

with st.sidebar:
    st.header("Local models")
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
    chat_model = st.text_input("Chat model", value="qwen2.5:7b")
    embedding_model = st.text_input("Embedding model", value="nomic-embed-text")
    chunk_size = st.slider(
        "Chunk size", min_value=300, max_value=2000, value=1000, step=100
    )
    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=min(500, chunk_size - 1),
        value=min(150, chunk_size - 1),
        step=50,
    )
    result_count = st.slider("Relevant chunks", min_value=1, max_value=8, value=4)
    st.caption(
        "Ollama must be running, with both models downloaded. "
        "For example: `ollama pull qwen2.5:7b` and "
        "`ollama pull nomic-embed-text`."
    )

uploaded_file = st.file_uploader(
    "Choose a document",
    type=["pdf", "docx", "txt", "md"],
    max_upload_size=50,
)
index_settings = (ollama_url, embedding_model, chunk_size, chunk_overlap)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if st.button("Index document", type="primary"):
        try:
            source_documents = extract_text(uploaded_file.name, file_bytes)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = splitter.split_documents(source_documents)
            embeddings = OllamaEmbeddings(
                model=embedding_model,
                base_url=ollama_url,
            )
            st.session_state.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
            )
            st.session_state.indexed_file_hash = file_hash
            st.session_state.indexed_file_name = uploaded_file.name
            st.session_state.indexed_settings = index_settings
            st.session_state.chat_history = []
            st.success(f"Indexed {uploaded_file.name} into {len(chunks)} text chunks.")
        except Exception as error:
            st.error(f"Could not index the document: {error}")

indexed_hash = st.session_state.get("indexed_file_hash")
is_indexed = (
    uploaded_file is not None
    and indexed_hash == file_hash
    and st.session_state.get("indexed_settings") == index_settings
)
if uploaded_file is not None and not is_indexed:
    st.info(
        "Index this document before asking questions. "
        "Upload or model-setting changes require re-indexing."
    )

if is_indexed:
    st.caption(f"Ready to answer questions about **{uploaded_file.name}**.")
    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the document")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the document and generating an answer..."):
                try:
                    vector_store = st.session_state.vector_store
                    relevant_documents = vector_store.similarity_search(
                        question,
                        k=result_count,
                    )
                    context = format_context(relevant_documents)
                    prompt = (
                        "Answer the question using only the document excerpts below. "
                        "If the excerpts do not contain the answer, say you could not "
                        "find it in the document. Cite supporting sources using the "
                        "source labels provided.\n\n"
                        f"Document excerpts:\n{context}\n\n"
                        f"Question: {question}"
                    )
                    response = ChatOllama(
                        model=chat_model,
                        base_url=ollama_url,
                        temperature=0,
                    ).invoke(prompt)
                    answer = str(response.content)
                    st.markdown(answer)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as error:
                    st.error(f"Could not answer the question: {error}")
