from dotenv import load_dotenv

from rag_store import (
    create_retriever,
    format_search_results,
    load_vectorstore,
)

from rag_chain import ask_rag


# ─────────────────────────────────────────────────────────────
# 환경변수 로드
# ─────────────────────────────────────────────────────────────
load_dotenv()


# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"

COLLECTION_NAME = "whisper_transcripts"


# ─────────────────────────────────────────────────────────────
# STEP 5: Retriever 생성
# ─────────────────────────────────────────────────────────────
vectorstore = load_vectorstore(
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME,
)


retriever = create_retriever(
    vectorstore=vectorstore,
    k=3,
)

print("✅ Retriever 생성 완료\n")

# ─────────────────────────────────────────────────────────────
# STEP 6: 검색 테스트
# ─────────────────────────────────────────────────────────────
query = "동부 주차 구조물에서의 침입 사건에 대해 서술하시오."

print("query :", query)

print("=" * 60)
print("🔍 검색 테스트")
print("=" * 60)

retrieved_docs = retriever.invoke(query)

print(format_search_results(retrieved_docs))
print()


# ─────────────────────────────────────────────────────────────
# STEP 7: RAG 답변 생성
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("🤖 RAG 답변")
print("=" * 60)

answer = ask_rag(
    question=query,
    retriever=retriever,
)

print(answer)
