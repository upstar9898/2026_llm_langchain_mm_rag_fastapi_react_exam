import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()


def load_vector_db() -> Chroma:
    """
    04_image_indexing.py 에서 persist_directory 로 저장한
    Vector DB를 디스크에서 불러온다.

    인덱싱(04)은 최초 1회만 실행하면 된다.
    이후 검색(05)은 저장된 DB를 불러와서 바로 사용한다.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")
    )
    return Chroma(persist_directory="./image_vector_db", embedding_function=embeddings)


def search_images(vector_db: Chroma, query: str, k: int = 3) -> list[dict]:
    """
    텍스트 쿼리로 유사한 이미지를 검색한다.

    내부 동작:
        1. query  →  Embedding  →  쿼리 벡터 (1536차원)
        2. 저장된 모든 캡션 벡터와 거리(distance) 계산
        3. 거리 짧은 순으로 정렬  →  상위 k개 반환

    distance(거리) vs 코사인 유사도:
        ChromaDB 기본값은 L2(유클리드) 거리를 사용한다.
        코사인 유사도와 반대로 숫자가 작을수록 더 유사하다.
            0.0  →  완전히 같음
            1.0+ →  많이 다름

    Args:
        query : 찾고 싶은 이미지를 설명하는 자연어 텍스트
        k     : 반환할 결과 수
    Returns:
        [{"rank", "file_name", "image_path", "caption", "distance"}, ...]
    """
    raw = vector_db.similarity_search_with_score(query=query, k=k)
    results = []
    for rank, (doc, dist) in enumerate(raw, start=1):
        results.append(
            {
                "rank": rank,
                "file_name": doc.metadata["file_name"],
                "image_path": doc.metadata["image_path"],
                "caption": doc.page_content,
                "distance": dist,
            }
        )

    return results

def print_results(query: str, results: list[dict]) -> None:
    print(f"\n{'='*50}")
    print(f"🔍 검색어: '{query}'")
    print(f"{'='*50}")
    for r in results:
        print(f"\n  [{r['rank']}위]  {r['file_name']}")
        print(f"  캡션  : {r['caption']}")
        print(f"  거리  : {r['distance']}  (작을수록 유사)")
        print(f"  경로  : {r['image_path']}")


# ── 실행 ──────────────────────────────────────────────────────
print("Vector DB 로드 중...")
vector_db = load_vector_db()

queries = [
    "주황색 고양이",
    "야외에서 뛰는 동물",
    "파란색 새",
    "실내에 있는 동물",
]

for query in queries:
    results = search_images(vector_db, query, k=2)
    print_results(query, results)
