# rag_chain.py

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


RAG_PROMPT = PromptTemplate.from_template(
    """
당신은 음성 전사 데이터의 요약을 바탕으로 질문에 답하는 RAG 어시스턴트입니다.
아래 검색된 내용을 참고해서 사용자 질문에 답하세요.

규칙:
- 검색된 내용에 근거해서 답변하세요.
- 근거가 부족하면 부족하다고 말하세요.
- 가능한 경우 어떤 파일에서 나온 내용인지 언급하세요.
- 추측은 최소화하세요.

[검색된 내용]
{context}

[사용자 질문]
{question}

[답변]
"""
)


def format_docs(docs: List[Document]) -> str:
    formatted = []

    for doc in docs:
        meta = doc.metadata

        formatted.append(
            f"[파일명] {meta.get('source', 'unknown')}\n"
            f"[생성일] {meta.get('created_at', 'unknown')}\n"
            f"[언어] {meta.get('language', 'unknown')}\n"
            f"[길이] {meta.get('duration', 'unknown')}초\n"
            f"[주제] {meta.get('topic', 'unknown')}\n\n"
            f"{doc.page_content}"
        )

    return "\n\n----\n\n".join(formatted)


def create_rag_chain(
    retriever,
    model: str = "gpt-4o-mini",
    temperature: float = 0,
):
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_rag(
    question: str,
    retriever,
    model: str = "gpt-4o-mini",
    temperature: float = 0,
) -> str:
    rag_chain = create_rag_chain(
        retriever=retriever,
        model=model,
        temperature=temperature,
    )

    return rag_chain.invoke(question)