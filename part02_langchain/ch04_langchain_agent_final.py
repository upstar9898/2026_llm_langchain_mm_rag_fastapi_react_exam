# ─────────────────────────────────────────────────────────────

# 예제 4 (실제 LLM 버전): ChatOpenAI + bind_tools Agent

# ─────────────────────────────────────────────────────────────

#

# 이 예제의 목표:

#   Mock Agent처럼 개발자가 직접 if문으로 Tool을 고르는 것이 아니라,
#   실제 LLM이 사용자의 자연어 질문을 읽고
#   어떤 Tool을 사용할지 스스로 판단하게 만드는 것입니다.

# 핵심 흐름:

#   1. 사용자가 질문한다.
#   2. LLM이 질문을 읽고 Tool 사용 여부를 판단한다.
#   3. Tool이 필요하면 AIMessage.tool_calls에 호출 정보가 담긴다.
#   4. 파이썬 코드가 실제 Tool 함수를 실행한다.
#   5. Tool 실행 결과를 ToolMessage로 다시 LLM에게 전달한다.
#   6. LLM이 Tool 결과를 읽고 최종 자연어 답변을 만든다.


# 실행 전 준비:

#   pip install langchain-openai langchain-core python-dotenv

# .env 파일 예:
#   OPENAI_API_KEY=sk-...

# Mock 버전과 비교했을 때 달라진 점:
#  LLM의 tool_callls 판단으로 호출할 tool을 결정하는 부분
#  LLM의 tool_calls 판단으로 대체
#  Tool 함수 자체는 그대로 재사용 가능

# ─────────────────────────────────────────────────────────────
# json:
#   Python 객체(dict, list)를 JSON 문자열로 바꾸거나,
#   JSON 문자열을 다시 Python 객체로 바꿀 때 사용합니다.

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

# model:
#   사용할 OpenAI 모델 이름입니다.
# temperature:
#   답변의 랜덤성 정도입니다.
#   0에 가까울수록 일관적인 답변을 합니다.

# Tool 선택은 매번 흔들리면 수업 시연이 어려우므로 temperature=0을 권장합니다.

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# LLM에게 제공할 tool 목록

tool_list = ["get_risk_summary", "count_objects_in_zone", "filter_danger_frames"]

# LLM의 tool_calls에는 Tool 이름이 문자열로 들어옵니다.
# 예:
#   "name": "get_risk_summary"
# 그러므로 Tool 이름으로 실제 Tool 객체를 빠르게 찾을 수 있도록
# 딕셔너리를 만들어 둡니다.
tools_map = {
    t.name : t for t in tool_list
}

# LLM에 Tool 등록 : bind_tools()
# bind_tools() : LLM에게 사용 가능한 Tool 목록을 알려주는 함수
llm_with_tools = llm.bind_tools(tool_list)

# ─────────────────────────────────────────────────────────────
# 2. Tool 함수 정의
# ─────────────────────────────────────────────────────────────
#
# 이 예제에서는 CCTV 분석 Agent가 사용할 Tool 3개를 정의합니다.
#
# Tool 1:
#   전체 위험도 요약
#
# Tool 2:
#   특정 구역 객체 탐지 수 집계
#
# Tool 3:
#   위험 프레임 목록 필터링
#
# 중요한 구분:
#   results_json:
#       risk_level, reason, action 등 분석 결과가 들어 있습니다.
#
#   frames_json:
#       location, detections, bbox 등 원본 탐지 결과가 들어 있습니다.
@tool
def get_risk_summary(results_json: str) -> str:
    """
    CCTV 분석 결과 전체를 요약합니다.
    위험/주의/정상 건수와 위험 프레임 ID 목록을 반환합니다.
    운영자가 "전체 요약", "위험 몇 건" 등을 물을 때 사용하세요.
    Args:
        results_json: risk_level이 포함된 CCTV 분석 결과 JSON 문자열
    Returns:
        위험/주의/정상 건수와 위험 프레임 ID 목록을 담은 JSON 문자열
    """
    # JSON 문자열을 Python 리스트로 변환합니다.
    data = json.loads(results_json)
    # 위험도별 개수를 저장할 딕셔너리입니다.
    counts = {
        "위험": 0,
        "주의": 0,
        "정상": 0,
    }
    # 위험 등급 프레임 ID만 따로 저장합니다.
    danger_ids = []
    # 분석 결과를 하나씩 확인합니다.
    for r in data:
        # risk_level 값이 없으면 기본값으로 "정상"을 사용합니다.
        lv = r.get("risk_level", "정상")
        # 해당 위험도 개수를 1 증가시킵니다.
        counts[lv] = counts.get(lv, 0) + 1
        # 위험 등급이면 frame_id를 따로 기록합니다.
        if lv == "위험":
            danger_ids.append(r["frame_id"])
    # 결과를 JSON 문자열로 반환합니다.
    # ensure_ascii=False를 사용해야 한글이 깨지지 않습니다.
    return json.dumps(
        {
            **counts,  # 딕셔너리에 들어있는 item을 풀어서 넣어줌
            "위험_프레임_ids": danger_ids,
        },
        ensure_ascii=False,
    )


@tool
def count_objects_in_zone(frames_json: str, zone: str) -> str:
    """
    특정 구역(zone)에서 탐지된 객체 수를 집계합니다.
    "창고 출입구 탐지 건수", "주차장 A 인원" 등의 질문에 사용하세요.
    zone 파라미터에는 구역 이름을 정확히 입력하세요.
    Args:
        frames_json: location, detections 정보가 포함된 원본 프레임 JSON 문자열
        zone: 집계할 구역 이름
    Returns:
        해당 구역의 프레임 수, 전체 탐지 수, 프레임당 평균 탐지 수 JSON 문자열
    """
    # JSON 문자열을 Python 리스트로 변환합니다.
    data = json.loads(frames_json)
    # 사용자가 요청한 구역과 location이 같은 프레임만 모읍니다.
    zone_frames = [f for f in data if f["location"] == zone]
    # 해당 구역 프레임들에서 탐지된 객체 수를 모두 더합니다.
    total = sum(len(f["detections"]) for f in zone_frames)
    # 프레임당 평균 탐지 수를 계산합니다.
    # zone_frames가 비어 있으면 0으로 나누는 오류를 막기 위해 0을 반환합니다.
    avg = round(total / len(zone_frames), 1) if zone_frames else 0
    # 집계 결과를 JSON 문자열로 반환합니다.
    return json.dumps(
        {
            "zone": zone,
            "frame_count": len(zone_frames),
            "total_detections": total,
            "avg_per_frame": avg,
        },
        ensure_ascii=False,
    )


@tool
def filter_danger_frames(results_json: str) -> str:
    """
    위험(risk_level = '위험') 등급 프레임만 필터링해서 반환합니다.
    "위험 프레임 목록", "위험한 것만 뽑아줘" 등의 질문에 사용하세요.
    Args:
        results_json: risk_level이 포함된 CCTV 분석 결과 JSON 문자열
    Returns:
        위험 등급 프레임 목록 JSON 문자열
    """
    # frames_json:
    #   location, detections, bbox 중심 데이터
    #
    # results_json:
    #   risk_level, reason, action 중심 데이터
    #
    # 따라서 위험 프레임을 필터링할 때는 results_json을 받아야 합니다.
    data = json.loads(results_json)
    # risk_level이 "위험"인 프레임만 추립니다.
    danger = [r for r in data if r.get("risk_level") == "위험"]
    # 위험 프레임 목록을 JSON 문자열로 반환합니다.
    return json.dumps(
        danger,
        ensure_ascii=False,
    )


class CCTVLLMAgent:
    """

    실제 ChatOpenAI를 사용하는 CCTV 분석 Agent입니다.
    이 Agent는 다음 작업을 수행합니다.

    1. 사용자 질문을 받습니다.
    2. LLM에게 질문과 분석 데이터를 전달합니다.
    3. LLM이 Tool 사용 여부를 판단합니다.   (Thought)
    4. Tool이 필요하면 실제 Tool을 실행합니다.  (Action)
    5. Tool 결과를 다시 LLM에게 전달합니다. (Observation)
    6. LLM이 최종 자연어 답변을 생성합니다.

    Mock Agent와 비교:
        Mock Agent:
            개발자가 if문으로 Tool을 선택합니다.
        LLM Agent:
            LLM이 자연어 질문을 읽고 Tool을 선택합니다.
    """

    # LLM에게 알려줄 역할과 Tool 선택 기준입니다.

    SYSTEM_PROMPT = """당신은 CCTV 보안 분석 AI 어시스턴트입니다.
운영자의 질문에 답변하기 위해 제공된 Tool을 적절히 활용하세요.
데이터 구분:
- results_json: risk_level, reason, action 등 위험도 분석 결과
- frames_json : location, detections, bbox 등 원본 탐지 결과
Tool 선택 기준:
- 전체 요약, 위험 건수 질문 → get_risk_summary
- 특정 구역 탐지 현황 질문 → count_objects_in_zone
- 위험 프레임 목록 질문   → filter_danger_frames
중요:
- 위험도, 위험 프레임, risk_level 관련 질문에는 results_json을 사용하세요.
- 구역, 위치, detections, bbox 관련 질문에는 frames_json을 사용하세요.
답변은 한국어로, 핵심 수치를 포함해 명확하게 작성하세요."""

    def __init__(self, results_json, frames_json):
        """
        Agent 생성자(초기화 함수)입니다.
        Args:

            results_json:
                위험도 분석 결과 JSON 문자열입니다.
                risk_level, reason, action 등이 들어 있습니다.

            frames_json:
                원본 탐지 결과 JSON 문자열입니다.
                location, detections, bbox 등이 들어 있습니다.
        """
        self.results_json = results_json
        self.frames_json = frames_json

        # Agent의 실행 이력을 저장하는 공간 (디버깅 용도)
        # 각 질문마다 (query, tool_calls, answer) 등의 정보를 기록
        self.scratchpad = []

    def run(self, query: str) -> str:
        """
        Agent 실행 메서드입니다.
        Args:
            query:
                사용자의 자연어 질문
        Returns:
            LLM이 생성한 최종 답변 문자열
        """
        print(f"\n  📌 질문: {query}")
        # ─────────────────────────────────────────────────────
        # 1단계: 메시지 구성 (LLM에게 전달할 프롬프트 조립)
        # ─────────────────────────────────────────────────────
        # SystemMessage:
        #   LLM의 역할과 규칙을 알려줍니다.

        # HumanMessage:
        #   사용자의 질문과 분석 데이터를 함께 전달합니다.

        # 데이터를 함께 넣는 이유:
        #   LLM이 Tool을 호출할 때 필요한 인자를 구성해야 하기 때문입니다.
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(
                content=f"{query}\n\n"
                f"[분석데이터]\n"
                f"results_json : {self.results_json}\n"
                f"frames_json : {self.frames_json}"
            ),
        ]
        # ─────────────────────────────────────────────────────
        # 2단계: LLM 호출 - Tool 선택 단계
        # ─────────────────────────────────────────────────────
        #
        # 여기서 LLM은 사용자의 질문을 읽고 판단합니다.
        #

        # 가능한 결과:
        #   1. Tool이 필요 없다고 판단
        #      → response.content에 바로 답변 생성
        #
        #   2. Tool이 필요하다고 판단
        #      → response.tool_calls에 Tool 호출 요청 생성
        #
        # 주의:
        #   이 단계에서 Tool이 실제 실행된 것은 아닙니다.
        print("  💭 LLM 판단 중...", end=" ", flush=True)
        response = llm_with_tools.invoke(messages)
        print("완료")
        


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

    results_json = json.dumps(results, ensure_ascii=False)
    frames_json = json.dumps(frames, ensure_ascii=False)

    agent = CCTVLLMAgent(results_json=results_json, frames_json=frames_json)
