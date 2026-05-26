# rag_store.py

from datetime import datetime
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def create_transcript_document(
    transcript: Dict[str, Any],
    topic: str,
    summary: str,
    created_at: str | None = None,
) -> Document:
    """
    Whisper 전사 결과의 요약 1개를 RAG용 Document 1개로 변환합니다.
    """

    if created_at is None:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    source = transcript.get("source", "unknown")
    duration = float(transcript.get("duration", 0.0))
    language = transcript.get("language", "unknown")

    page_content = f"""[주제]
{topic}

[요약]
{summary}
"""

    metadata = {
        "source": source,
        "duration": duration,
        "created_at": created_at,
        "language": language,
        "topic": topic,
    }

    return Document(
        page_content=page_content,
        metadata=metadata,
    )


def create_vectorstore_from_documents(
    documents: List[Document],
    persist_directory: str = "./chroma_whisper_db",
    collection_name: str = "whisper_transcripts",
    embedding_model: str = "text-embedding-3-small",
) -> Chroma:
    """
    Document 리스트를 ChromaDB에 임베딩하여 저장합니다.
    """

    embeddings = OpenAIEmbeddings(model=embedding_model)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    return vectorstore


def load_vectorstore(
    persist_directory: str = "./chroma_whisper_db",
    collection_name: str = "whisper_transcripts",
    embedding_model: str = "text-embedding-3-small",
) -> Chroma:
    """
    기존 ChromaDB를 불러옵니다.
    """

    embeddings = OpenAIEmbeddings(model=embedding_model)

    vectorstore = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    return vectorstore


def add_transcript_to_vectorstore(
    transcript: Dict[str, Any],
    topic: str,
    summary: str,
    persist_directory: str = "./chroma_whisper_db",
    collection_name: str = "whisper_transcripts",
    embedding_model: str = "text-embedding-3-small",
) -> Chroma:
    """
    Whisper 전사 결과를 Document로 만들고 ChromaDB에 추가합니다.
    """

    document = create_transcript_document(
        transcript=transcript,
        topic=topic,
        summary=summary,
    )

    vectorstore = load_vectorstore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    vectorstore.add_documents([document])

    return vectorstore


def create_retriever(
    vectorstore: Chroma,
    k: int = 3,
):
    """
    ChromaDB를 Retriever로 변환합니다.
    """

    return vectorstore.as_retriever(search_kwargs={"k": k})


def format_search_results(docs: List[Document]) -> str:
    """
    검색 결과 확인용 문자열 생성.
    """

    results = []

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata

        results.append(
            f"[{i}] "
            f"source={meta.get('source')} | "
            f"duration={meta.get('duration')} | "
            f"created_at={meta.get('created_at')} | "
            f"language={meta.get('language')} | "
            f"topic={meta.get('topic')}\n"
            f"{doc.page_content[:300]}"
        )

    return "\n\n".join(results)
