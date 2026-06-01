## 이미지 RAG란 무엇인가

# 일반 RAG는 텍스트 문서를 벡터로 저장하고 검색한다.
# **이미지 RAG** 는 이미지의 **설명(캡션)** 을 벡터로 저장하고 검색한다.

# ```
# [사전 작업 — Indexing]
#   이미지 파일
#       └→ Vision API → 이미지 설명(캡션) 텍스트
#                           └→ Embedding → 벡터
#                                             └→ Vector DB 저장
#                                                (캡션 텍스트 + 이미지 파일 경로)

# [검색 — Retrieval]
#   검색 쿼리 (예: "주황색 고양이")
#       └→ Embedding → 쿼리 벡터
#                          └→ Vector DB 유사도 비교 → 유사한 이미지 Top-K 반환
#                                                        └→ metadata['image_path'] 로 파일 복원
# ```

# 핵심은 단순하다.
# 이미지를 직접 벡터로 저장하는 게 아니라, **Vision API가 만든 텍스트 설명을 벡터로 저장** 한다.
# 그래서 "고양이가 있는 사진 찾아줘" 같은 텍스트 쿼리로 이미지를 찾을 수 있다.

import json, os, re, base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from ch04_01_visionModel_basic import base64_to_image
from ch04_02_vision_api_call import json_parse, analyze_image
import chromadb

import datetime

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def index_image_folder(folder_path: str) -> chromadb:
    """
    폴더 안의 모든 이미지 파일을 Vision API로 분석해서
    Vector DB에 인덱싱한다.

    처리 흐름:
        이미지 파일 목록 탐색
            └→ 각 이미지 → generate_caption() → 캡션 텍스트
                                └→ Document(page_content=캡션, metadata={"image_path": ...})
                                        └→ Chroma.from_documents() → Vector DB

    Returns:
        검색 가능한 Chroma Vector DB 객체
    """
    # 지원 확장자 필터
    supported = {".jpg", ".jpeg", ".png", ".webp"}

    prompt = """이 이미지를 분석해서 아래 JSON 형식으로만 응답하세요.
    {
    "subject": "이미지의 주요 피사체",
    "description": "전체적인 설명 (2~3문장)",
    "mood": "이미지의 분위기",
    "colors": ["주요 색상1", "주요 색상2"]
    }"""
    image_files = [
        p for p in Path(folder_path).iterdir() if p.suffix.lower() in supported
    ]

    if not image_files:
        raise ValueError(f"{folder_path}에 이미지 파일이 없습니다.")
    print(f"총 {len(image_files)}개 이미지 발견")

    documents = []
    for i, img_path in enumerate(image_files, start=1):
        print(f"  [{i}/{len(image_files)}] {img_path.name} → 캡션 생성 중...")
        caption_dict = analyze_image(str(img_path), prompt=prompt)
        # Document 객체 생성
        documents.append(
            Document(
                page_content=caption_dict.get("description", ""),
                metadata={
                    "image_path": str(img_path),
                    "file_name": img_path.name,
                    "created_at": str(datetime.datetime.now()),
                },
            )
        )

    # embedding
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory="./image_vector_db",
    )

    return vector_db


if __name__ == "__main__":
    # "./vision_sample" 폴더 안의 모든 이미지를 인덱싱
    index_image_folder("./vision_sample")
