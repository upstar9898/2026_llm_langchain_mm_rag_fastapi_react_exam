# **파일명:** `ch03_02_multi_query_retriever.py`
# > **왜 필요한가?**
# > "새벽 창고 침입"을 검색했을 때, "심야 창고 무단 접근"이라고 저장된 유사 사례가 누락될 수 있습니다.
# > GPT-4o가 같은 뜻을 여러 표현으로 재작성해서 더 다양한 결과를 가져옵니다.

import sys
import logging
from dotenv import load_dotenv
from typing import List

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

# ─── 문서 준비 ────────────────────────────────────────────────
docs = [
    Document(
        page_content="새벽 2시 31분 창고 출입구 침입. person:2 car:1 탐지. 위험. 경보 후 도주."
    ),
    Document(
        page_content="새벽 3시 15분 공장 외곽 침입 시도. person:3 car:2 탐지. 위험. 경찰 출동."
    ),
    Document(page_content="오후 2시 주차장 A. person:1 탐지. 정상 이용객."),
    Document(
        page_content="심야 창고 주변 배회. person:1 반복 배회 감지. 주의. 경비 순찰."
    ),
    Document(
        page_content="새벽 1시 주차장 B 침입 시도. person:2 탐지. 차량 접근 반복. 위험."
    ),
    Document(
        page_content="새벽 창고 출입구 야간 무단 접근. 차량 1대 정차 후 2인 하차. 경비 즉시 출동."
    ),
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ─────────────────────────────────────────────────────────────
# MultiQueryRetriever 구성
#
# [작동 원리]
# 1. 원래 질문 → GPT-4o가 3가지 다른 표현으로 재작성
#    예) "새벽 창고 침입 의심"
#     → "야간 창고 출입구 무단 접근 사례"
#     → "심야 시간대 창고 주변 복수 인원 탐지"
#     → "새벽 차량 동반 침입 패턴"
# 2. 원래 + 재작성 3개 = 총 4가지 쿼리로 각각 검색
# 3. 중복 제거 후 더 다양한 관련 문서 반환
#
# temperature=0.3: 약간의 창의성으로 다양한 표현 생성
# ───────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# 기본 설정으로 로그를 남길 수 있는 로거객체를 가져와 langchain_classic.retriever.multi_query의 이름으로 하고, 기본 로그레벨을 logging.INFO 레벨로 지정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
    force=True,  # ← 기존 핸들러 강제 초기화 후 재설정
)
logging.getLogger("langchain_classic.retriever.multi_query").setLevel(logging.INFO)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)

query = "새벽 창고 출입구 침입 의심. person:2 / car:1 탐지."

print("=" * 60)
print("📌 기본 Retriever vs MultiQueryRetriever 비교")
print("=" * 60)

# 기본 Retriever 결과
basic_docs = base_retriever.invoke(query)
print(f"\n[기본 Retriever] → {len(basic_docs)}개:")
for i, d in enumerate(basic_docs, 1):
    print(f"  [{i}] {d.page_content[:65]}")

multi_docs = multi_retriever.invoke(query)

print(f"→ {len(multi_docs)}개 (중복 제거 후):")
for i, d in enumerate(multi_docs, 1):
    print(f"  [{i}] {d.page_content[:65]}")

print(f"\n✅ 기본: {len(basic_docs)}개 → MultiQuery: {len(multi_docs)}개")


# multi_retriever로 검색된 다양한 응답의 내용을 랭체인을 이용하여 llm에게 주고, 종합적인 응답을 받도록 한다

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────────────────────
# LLM 종합 분석
# multi_retriever로 검색된 다양한 문서들을 컨텍스트로 삼아
# GPT-4o에게 현재 상황에 대한 종합 판단을 요청합니다.
#
# [흐름]
# multi_docs (검색 결과) → format_docs() → 텍스트로 합치기
#                                               ↓
#                          PromptTemplate → {context} + {question}
#                                               ↓
#                                           ChatOpenAI (GPT-4o)
#                                               ↓
#                                         StrOutputParser → 문자열
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("🤖 MultiQuery 검색 결과 → GPT-4o 종합 분석")
print("=" * 60)


def format_docs(docs: List[Document]) -> str:
    """
    검색된 Document 리스트를 프롬프트에 넣을 텍스트로 변환합니다.
    번호를 붙여서 GPT-4o가 각 사례를 구분하기 쉽게 합니다.
    """
    return "\n\n".join(
        f"[사례 {i}] {doc.page_content}" for i, doc in enumerate(docs, 1)
    )


# 프롬프트 정의
# {context}: format_docs()로 만든 검색 결과 텍스트
# {question}: 현재 탐지 상황 (원래 쿼리)
prompt = PromptTemplate.from_template(
    """당신은 CCTV 보안 분석 전문가입니다.
아래 검색된 과거 탐지 사례들을 종합하여 현재 상황에 대한 분석과 대응 방안을 제시하세요.
 
[검색된 유사 사례]
{context}
 
[현재 탐지 상황]
{question}
 
위험도(정상/주의/위험), 판단 근거, 권장 조치사항을 구체적으로 답변하세요."""
)

# LCEL 체인 구성
# multi_docs는 이미 검색이 끝난 상태이므로
# Retriever를 체인에 연결하는 대신 format_docs()로 직접 변환해서 넣습니다.
analysis_chain = prompt | llm | StrOutputParser()

# 검색 결과를 컨텍스트로 변환 후 LLM 호출
context_text = format_docs(multi_docs)

print(f"\n📋 컨텍스트로 사용할 사례 수: {len(multi_docs)}개")
print(f"🔍 분석 대상 상황: {query}\n")

answer = analysis_chain.invoke(
    {
        "context": context_text,  # MultiQuery로 검색된 다양한 사례들
        "question": query,  # 현재 탐지 상황
    }
)

print("📊 GPT-4o 종합 분석 결과:")
print(answer)
