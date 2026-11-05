import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from pinecone import Pinecone

load_dotenv()

CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:1.7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")
PINECONE_TEXT_FIELD = os.getenv("PINECONE_TEXT_FIELD", "chunk_text")
PINECONE_API_VERSION = "2026-07"


def load_documents() -> list[Document]:
    source = Path(__file__).resolve().with_name("vectordb.txt")
    return [
        Document(
            page_content=source.read_text(encoding="utf-8"),
            metadata={"source": str(source)},
        )
    ]


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def get_required_setting(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} in your environment or .env file.")
    return value


def get_pinecone_index() -> tuple[Any, str]:
    client = Pinecone(api_key=get_required_setting("PINECONE_API_KEY"))
    description = client.describe_index(get_required_setting("PINECONE_INDEX_NAME"))
    return client.Index(host=description.host), description.host


def pinecone_documents_request(
    host: str, operation: str, payload: dict[str, Any]
) -> dict[str, Any]:
    namespace = quote(PINECONE_NAMESPACE or "__default__", safe="")
    request = Request(
        f"https://{host}/namespaces/{namespace}/documents/{operation}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Api-Key": get_required_setting("PINECONE_API_KEY"),
            "Content-Type": "application/json",
            "X-Pinecone-Api-Version": PINECONE_API_VERSION,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Pinecone Documents API {operation} failed "
            f"({error.code}): {response_body}"
        ) from error


@traceable
def index_documents() -> int:
    chunks = split_documents(load_documents())
    source = Path(__file__).resolve().with_name("vectordb.txt")
    documents = [
        {
            "_id": hashlib.sha256(
                f"{source}:{chunk.metadata.get('start_index', index)}:"
                f"{chunk.page_content}".encode("utf-8")
            ).hexdigest(),
            PINECONE_TEXT_FIELD: chunk.page_content,
            "source": str(source),
            "title": source.name,
        }
        for index, chunk in enumerate(chunks)
    ]
    index, host = get_pinecone_index()
    namespace = PINECONE_NAMESPACE or "__default__"
    namespaces = index.describe_index_stats().namespaces or {}
    if namespace in namespaces:
        pinecone_documents_request(
            host,
            "delete",
            {"filter": {"source": {"$eq": str(source)}}},
        )
    for start in range(0, len(documents), 96):
        pinecone_documents_request(
            host,
            "upsert",
            {"documents": documents[start : start + 96]},
        )
    return len(chunks)


def search_documents(question: str, top_k: int = 4) -> list[Document]:
    _, host = get_pinecone_index()
    result = pinecone_documents_request(
        host,
        "search",
        {
            "top_k": top_k,
            "score_by": [
                {
                    "type": "embed",
                    "fields": [PINECONE_TEXT_FIELD],
                    "query": question,
                }
            ],
            "include_fields": [PINECONE_TEXT_FIELD, "source"],
        },
    )
    matches = result.get("matches", [])
    if not isinstance(matches, list):
        raise RuntimeError("Pinecone Documents API returned an invalid matches list.")

    documents = []
    for match in matches:
        if not isinstance(match, dict):
            raise RuntimeError(
                "Pinecone Documents API returned an invalid search match."
            )
        page_content = match.get(PINECONE_TEXT_FIELD)
        source = match.get("source")
        if not isinstance(page_content, str) or not isinstance(source, str):
            raise RuntimeError(
                "Pinecone search result is missing the configured text field "
                f"{PINECONE_TEXT_FIELD!r} or its source metadata."
            )
        documents.append(
            Document(page_content=page_content, metadata={"source": source})
        )
    return documents


@traceable
def answer_question(question: str) -> str:
    if not question.strip():
        raise ValueError("Question must not be empty.")

    documents = search_documents(question)
    if not documents:
        return "I couldn't find any relevant documents in the Pinecone index."

    context = "\n\n".join(
        f"Source: {Path(document.metadata['source']).name}\n" f"{document.page_content}"
        for document in documents
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question using only the supplied context. "
                "If the context does not contain the answer, say you do not know. "
                "Cite the source filename in your answer.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    chain = (
        prompt
        | ChatOllama(
            model=CHAT_MODEL,
            temperature=0,
            base_url=OLLAMA_BASE_URL,
        )
        | StrOutputParser()
    )
    return chain.invoke({"context": context, "question": question})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index and query vectordb.txt with RAG."
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser(
        "index", help="Split and upload vectordb.txt to Pinecone (the default)."
    )
    ask_parser = commands.add_parser("ask", help="Ask a question about vectordb.txt.")
    ask_parser.add_argument(
        "question", help="what is vertordb?", type=str, nargs="+"
    )
    return parser.parse_args()


@traceable
def main() -> None:
    args = parse_args()
    if args.command in (None, "index"):
        chunk_count = index_documents()
        print(f"Indexed {chunk_count} chunk(s) from vectordb.txt in Pinecone.")
    else:
        print(answer_question(args.question))


if __name__ == "__main__":
    os.environ.setdefault("LANGCHAIN_PROJECT", "langchain-learning")
    main()
