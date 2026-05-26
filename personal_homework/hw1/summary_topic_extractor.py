# summary_extractor.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


SUMMARY_PROMPT = PromptTemplate.from_template(
    """
당신은 음성 전사 데이터를 RAG 검색용으로 요약하는 도우미입니다.

아래 전사 원문을 보고, 검색에 도움이 되도록 핵심 내용을 2~3문장으로 요약하세요.

조건:
- 2~3문장으로 작성
- 중요한 사건, 장소, 대상, 행동, 판단을 포함
- 너무 추상적으로 요약하지 말 것
- 설명, 번호, 제목 없이 요약문만 출력

[전사 원문]
{transcript}
"""
)


def create_summary_extractor(
    model: str = "gpt-4o-mini",
    temperature: float = 0,
):
    llm = ChatOpenAI(model=model, temperature=temperature)

    summary_chain = (
        SUMMARY_PROMPT
        | llm
        | StrOutputParser()
    )

    return summary_chain


def extract_summary(
    transcript_text: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0,
) -> str:
    if not transcript_text or not transcript_text.strip():
        return "내용 없음"

    summary_chain = create_summary_extractor(
        model=model,
        temperature=temperature,
    )

    summary = summary_chain.invoke({
        "transcript": transcript_text.strip()
    })

    return summary.strip()

TOPIC_PROMPT = PromptTemplate.from_template(
    """
당신은 음성 전사 데이터를 RAG 검색용으로 정리하는 도우미입니다.

아래 전사 원문을 보고, 대화의 핵심 주제를 한 문장으로 요약하세요.

조건:
- 1문장만 출력
- 너무 길게 쓰지 말 것
- 검색에 도움이 되는 핵심 키워드를 포함할 것
- 설명, 번호, 따옴표 없이 주제 문장만 출력

[전사 원문]
{transcript}
"""
)


def create_topic_extractor(
    model: str = "gpt-4o-mini",
    temperature: float = 0,
):
    llm = ChatOpenAI(model=model, temperature=temperature)

    topic_chain = TOPIC_PROMPT | llm | StrOutputParser()

    return topic_chain


def extract_topic(
    transcript_text: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0,
) -> str:
    if not transcript_text or not transcript_text.strip():
        return "내용 없음"

    topic_chain = create_topic_extractor(
        model=model,
        temperature=temperature,
    )

    topic = topic_chain.invoke({"transcript": transcript_text.strip()})

    return topic.strip()
