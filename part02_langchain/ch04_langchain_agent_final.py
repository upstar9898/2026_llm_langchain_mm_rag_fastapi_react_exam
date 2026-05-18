# ─────────────────────────────────────────────────────────────

# 예제 4 (실제 LLM 버전): ChatOpenAI + bind_tools Agent

# ─────────────────────────────────────────────────────────────

#

# 이 예제의 목표:

#   Mock Agent처럼 개발자가 직접 if문으로 Tool을 고르는 것이 아니라,

#   실제 LLM이 사용자의 자연어 질문을 읽고

#   어떤 Tool을 사용할지 스스로 판단하게 만드는 것입니다.

#

# 핵심 흐름:

#   1. 사용자가 질문한다.

#   2. LLM이 질문을 읽고 Tool 사용 여부를 판단한다.

#   3. Tool이 필요하면 AIMessage.tool_calls에 호출 정보가 담긴다.

#   4. 파이썬 코드가 실제 Tool 함수를 실행한다.

#   5. Tool 실행 결과를 ToolMessage로 다시 LLM에게 전달한다.

#   6. LLM이 Tool 결과를 읽고 최종 자연어 답변을 만든다.

#

# 실행 전 준비:

#   pip install langchain-openai langchain-core python-dotenv

#

# .env 파일 예:

#   OPENAI_API_KEY=sk-...

#

# Mock 버전과 비교했을 때 달라진 점:

#  LLM의 tool_callls 판단으로 호출할 tool을 결정하는 부분

#  LLM의 tool_calls 판단으로 대체

#  Tool 함수 자체는 그대로 재사용 가능

# ─────────────────────────────────────────────────────────────

# json:

#   Python 객체(dict, list)를 JSON 문자열로 바꾸거나,

#   JSON 문자열을 다시 Python 객체로 바꿀 때 사용합니다.

#

# load_dotenv:

#   .env 파일에 적어둔 환경변수를 현재 Python 실행 환경으로 불러옵니다.

import json

from dotenv import load_dotenv

# ChatOpenAI:

#   LangChain에서 OpenAI Chat 모델을 사용할 수 있게 해주는 클래스입니다.

from langchain_openai import ChatOpenAI

# tool:

#   일반 Python 함수를 LangChain Tool로 바꿔주는 데코레이터입니다.

#   @tool을 붙이면 함수 이름, docstring, 파라미터 정보를

#   LLM이 이해할 수 있는 Tool 스키마로 변환합니다.

from langchain_core.tools import tool

# LangChain 메시지 타입들:

# HumanMessage:

#   사용자의 질문을 나타내는 메시지입니다.

# ToolMessage:

#   Tool 실행 결과를 LLM에게 다시 전달할 때 사용하는 메시지입니다.

# SystemMessage:

#   LLM에게 역할, 규칙, 답변 방식 등을 알려주는 메시지입니다.

from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    SystemMessage,
)

# .env 파일에서 OPENAI_API_KEY를 읽어옵니다.

load_dotenv()

# ─────────────────────────────────────────────────────────────

# 1. LLM 초기화

# ─────────────────────────────────────────────────────────────

# ChatOpenAI 객체를 생성합니다.

#

# model:

#   사용할 OpenAI 모델 이름입니다.

#

# temperature:

#   답변의 랜덤성 정도입니다.

#   0에 가까울수록 일관적인 답변을 합니다.

#

# Tool 선택은 매번 흔들리면 수업 시연이 어려우므로

# temperature=0을 권장합니다.

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

if __name__ == "__main__":
    # 테스트 데이터를 만들기 위해 random 모듈을 사용합니다.

    import random

    # 랜덤 결과를 고정합니다.

    # 수업 시연에서는 실행할 때마다 결과가 바뀌지 않는 것이 좋습니다.

    random.seed(42)

    # CCTV 위치 목록입니다.

    LOCATIONS = [
        "주차장 A",
        "창고 출입구",
        "로비",
        "비상구 복도",
        "옥상",
    ]
    results = [
        {
            "frame_id": i,
            "risk_level": "위험" if i % 4 == 0 else ("주의" if i % 3 == 0 else "정상"),
            "person_count": random.randint(0, 4),
            "reason": "심야 다인 탐지" if i % 4 == 0 else "일반",
            "action": "경비팀 출동" if i % 4 == 0 else "이상 없음",
        }
        for i in range(1, 11)
    ]
    # ─────────────────────────────────────────────────────────

    # 원본 탐지 프레임 데이터 생성

    # ─────────────────────────────────────────────────────────

    # frames_json으로 변환될 데이터입니다.

    # 이 데이터에는 location, detections, bbox가 들어 있으므로,

    # 특정 구역 탐지 수 집계 Tool에서 사용합니다.

    frames = [
        {
            "frame_id": i,
            "timestamp": f"0{2 if i % 3 == 0 else 1}:{i % 60:02d}",
            "location": LOCATIONS[i % len(LOCATIONS)],
            "detections": [
                {
                    "class": "person",
                    "bbox": [10, 10, 100, 100],
                    "confidence": 0.91,
                }
                for _ in range(random.randint(0, 3))
            ],
        }
        for i in range(1, 11)
    ]

    result_json = json.dumps(results, ensure_ascii=False)
    frames_json = json.dumps(frames, ensure_ascii=False)