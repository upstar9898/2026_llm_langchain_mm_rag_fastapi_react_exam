# ─────────────────────────────────────────────────────────────
# 예제 3: ReAct 패턴 수동 시뮬레이션
#
# 질문: "위험 프레임 요약 + 창고 출입구 탐지 현황 같이 알려줘"
#
# LLM이 실제로 하는 일:
#   1. 질문 분석 → 어떤 Tool이 필요한지 판단 (Thought)
#   2. Tool 호출 (Action)
#   3. 결과 확인 (Observation)
#   4. 충분한지 판단 → 더 필요하면 다시 1번으로
#   5. 최종 답변 생성 (Final Answer)
# ─────────────────────────────────────────────────────────────

import json
from langchain_core.tools import tool
import json, re, random
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser

# ── 공통 데이터: 10개 프레임 시뮬레이션 ───────────────────────
LOCATIONS = ["주차장 A", "창고 출입구", "로비", "비상구 복도", "옥상"]


def make_frame_results(n: int = 10) -> list:
    """
    OpenCV가 영상에서 추출했을 프레임 탐지 결과를 시뮬레이션합니다.
    실제 수업에서는 OpenCV cv2.VideoCapture로 생성한 결과를 사용합니다.
    """
    frames = []
    for i in range(1, n + 1):
        n_persons = random.randint(0, 5)
        frames.append(
            {
                "frame_id": i,
                # 3의 배수 프레임은 새벽 시간 (02:xx) → 위험도 높음
                "timestamp": f"0{2 if i % 3 == 0 else 1}:{i // 60:02d}:{i % 60:02d}",
                "location": LOCATIONS[i % len(LOCATIONS)],
                "detections": [
                    {
                        "class": "person",
                        "bbox": [
                            random.randint(0, 400),
                            random.randint(0, 300),
                            random.randint(100, 500),
                            random.randint(100, 400),
                        ],
                        "confidence": round(random.uniform(0.75, 0.98), 2),
                    }
                    for _ in range(n_persons)
                ],
            }
        )
    return frames


def format_detections(frame: dict) -> str:
    """
    OpenCV 탐지 결과 딕셔너리 → LLM 읽기 좋은 텍스트로 변환.

    Part 01에서 만든 format_detections 함수를 그대로 재활용합니다.
    이처럼 Part 01 코드는 이 강의 전체에서 반복 사용됩니다.
    """
    lines = [f"[{frame['timestamp']}] {frame['location']}"]
    for d in frame["detections"]:
        lines.append(
            f"  - {d['class']} (신뢰도 {d['confidence']:.0%}), bbox={d['bbox']}"
        )
    if not frame["detections"]:
        lines.append("  - 탐지된 객체 없음")
    return "\n".join(lines)


# ── Mock LLM (API 키 불필요) ───────────────────────────────────
def mock_llm_fn(prompt_value) -> object:
    """
    실제 ChatOpenAI 대신 사용하는 Mock 함수.

    ChatOpenAI로 교체 시:
    ─────────────────────────────────────────────
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # 아래 mock_llm_fn 자리에 llm을 그대로 사용하면 됩니다
    analysis_chain = prompt | llm | json_parser
    ─────────────────────────────────────────────
    """
    # prompt가 렌더링한 메시지에서 human 내용 추출
    human_content = ""
    messages = prompt_value.to_messages()
    for m in messages:
        if hasattr(m, "content") and "frame_id" in str(m.content):
            human_content = str(m.content)
            break

    # frame_id 추출 (정규식)
    fid_match = re.search(r"frame_id:\\s*(\\d+)", human_content)
    frame_id = int(fid_match.group(1)) if fid_match else 0

    # 위험도 판단 로직 (실제 LLM 역할)
    person_count = human_content.count("person")
    has_night = bool(re.search(r"0[123]:\d{2}", human_content))  # 새벽 01~03시
    has_sensitive = "창고" in human_content or "비상구" in human_content

    if person_count >= 2 and has_night:
        risk, reason = "위험", f"심야 {person_count}인 탐지"
    elif person_count >= 1 and has_sensitive:
        risk, reason = "주의", "민감 구역 인원 탐지"
    elif person_count == 0:
        risk, reason = "정상", "탐지 객체 없음"
    else:
        risk, reason = "정상", "일반 탐지"

    response_text = json.dumps(
        {
            "frame_id": frame_id,
            "risk_level": risk,
            "person_count": person_count,
            "reason": reason,
            "action": (
                "경비팀 즉시 출동"
                if risk == "위험"
                else "모니터링 강화"
                if risk == "주의"
                else "이상 없음"
            ),
        },
        ensure_ascii=False,
    )

    class FakeResponse:
        """ChatOpenAI 응답 객체를 흉내 내는 클래스 (.content 속성 필요)"""

        def __init__(self, text):
            self.content = text

    return FakeResponse(response_text)


# ── LCEL 파이프라인 구성 ────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "CCTV 탐지 결과를 분석해 위험도를 JSON으로 반환하세요. "
            "risk_level은 '정상'/'주의'/'위험' 중 하나.",
        ),
        ("human", "frame_id: {frame_id}\\n탐지 결과:\\n{detection_text}"),
    ]
)

json_parser = JsonOutputParser()

# 파이프라인: prompt → mock_llm → .content 추출 → JSON 파싱
analysis_chain = (
    prompt
    | RunnableLambda(mock_llm_fn)  # Mock LLM (→ 실제는 ChatOpenAI)
    | RunnableLambda(lambda r: r.content)  # .content 문자열 추출
    | json_parser  # JSON 문자열 → dict
)


# Tool1 : 위험한 프레임을 필터링
@tool  # LLM스스로가 판단하여 이 함수를 호출 할 수 있도록...
def filter_danger_frames(frames_json: str) -> str:
    """
    분석 결과 리스트에서 위험(risk_level='위험') 프레임만 필터링합니다.

    CCTV 분석 결과에서 즉각 조치가 필요한 위험 이벤트만 추출할 때 사용합니다.

    Args:
        frames_json: 분석 결과 리스트를 JSON 문자열로 전달
                     예: '[{"frame_id":3,"risk_level":"위험",...}]'
    Returns:
        위험 프레임만 포함한 JSON 문자열 (빈 리스트 가능)
    """
    frames = json.loads(frames_json)  # JSON 파싱
    danger = [f for f in frames if f.get("risk_level") == "위험"]  # 위험만 필터링
    return json.dumps(danger, ensure_ascii=False, indent=2)


# ── Tool 2: 특정 구역 객체 카운트 ────────────────────────────
@tool
def count_objects_in_zone(frames_json: str, zone: str) -> str:
    """
    특정 구역(zone)에서 탐지된 객체 수를 카운트합니다.

    구역별 보안 밀도를 확인하거나 특정 위치의 이상 여부를 판단할 때 사용합니다.

    Args:
        frames_json: 프레임 탐지 결과 리스트 JSON 문자열
                     (frame_results의 원본 데이터 사용)
        zone: 카운트할 구역 이름 (예: '창고 출입구', '주차장 A', '로비')
    Returns:
        구역명, 총 탐지 수, 프레임 수, 평균을 담은 JSON 문자열
    """
    frames = json.loads(frames_json)

    # zone과 일치하는 프레임만 추출
    matched = [f for f in frames if f.get("location", "") == zone]

    # 해당 구역의 전체 탐지 객체 수 합산
    total = sum(len(f.get("detections", [])) for f in matched)

    return json.dumps(
        {
            "zone": zone,
            "total_detections": total,
            "frame_count": len(matched),
            # 프레임이 없으면 0 (ZeroDivisionError 방지)
            "avg_per_frame": round(total / len(matched), 2) if matched else 0,
        },
        ensure_ascii=False,
    )


# ── Tool 3: 전체 위험도 요약 ─────────────────────────────────
@tool
def get_risk_summary(results_json: str) -> str:
    """
    전체 분석 결과의 위험도 요약 통계를 반환합니다.

    운영자에게 현재 상황을 한눈에 보여줄 때 사용합니다.

    Args:
        results_json: 분석 결과 리스트 JSON 문자열
    Returns:
        정상/주의/위험 카운트와 위험 프레임 ID 목록을 담은 JSON 문자열
    """
    data = json.loads(results_json)
    summary = {"정상": 0, "주의": 0, "위험": 0, "위험_프레임_ids": []}

    for r in data:
        lvl = r.get("risk_level", "정상")
        summary[lvl] = summary.get(lvl, 0) + 1  # 위험도별 카운트
        if lvl == "위험":
            summary["위험_프레임_ids"].append(r.get("frame_id"))  # 위험 프레임 ID 수집

    return json.dumps(summary, ensure_ascii=False)


# 예제 2에서 만든 Tool 레지스트리 사용
# TOOL REGISTRY : Tool 이름 문자열을 실제 Tool객체와 연결해 둔 딕셔너리
TOOL_REGISTRY = {
    "filter_danger_frames": filter_danger_frames,
    "count_objects_in_zone": count_objects_in_zone,
    "get_risk_summary": get_risk_summary,
}


def react_step(
    thought: str,  # LLM 추론
    action_name: str,  # 호출해야할 tool이름
    action_input: dict,  # Tool에 전달할 파라미터
) -> str:
    """
    ReAct 한 사이클(Thought → Action → Observation)을 실행합니다.

    실제 Agent에서는 LLM이 thought와 action_name을 자동으로 생성하지만,
    여기서는 우리가 직접 입력해서 흐름을 눈으로 확인합니다.
    """
    print(f"\n  💭 Thought     : {thought}")
    print(f"  🔧 Action      : {action_name}({str(action_input)[:50]}...)")

    observation = TOOL_REGISTRY[action_name].invoke(action_input)

    # Observation 출력

    # 결과가 너무 길면 앞 80자만 출력한다.
    preview = (
        str(observation)[:80] + "..." if len(str(observation)) > 80 else observation
    )
    print(f" 👀 Observation : {preview}")

    # Observation 반환

    # 다음 단계에서 최종 답변을 만들 때 사용한다.

    return observation


print("[ReAct 시뮬레이션]")

print("질문: '위험 프레임 요약 + 창고 출입구 탐지 현황 같이 알려줘'\n")

print("사이클 1: 전체 요약 먼저")

# ── 사이클 1 ─────────────────────────────────────────────────

# 첫 번째 Thought:

# - 사용자의 질문에 "위험 프레임 요약"이 들어 있으므로

# - 전체 위험도 통계가 필요하다고 판단한다.

#

# 첫 번째 Action:

# - get_risk_summary Tool 실행

#

# 첫 번째 Observation:

# - 정상 / 주의 / 위험 개수와 위험 프레임 ID 목록

# ── 배치 실행 ────────────────────────────────────────────────
frame_results = make_frame_results(10)
print("배치 분석 실행 중...")

results = []
for frame in frame_results:
    result = analysis_chain.invoke(
        {
            "frame_id": frame["frame_id"],
            "detection_text": format_detections(frame),
        }
    )
    # frame_id 보정 (파이프라인 내에서 손실 방지)
    result["frame_id"] = frame["frame_id"]
    results.append(result)

results_json = json.dumps(results, ensure_ascii=False)


obs1 = react_step(
    thought="전체 위험도 분포를 먼저 파악해야겠다.",
    action_name="get_risk_summary",
    action_input={"results_json": results_json},
)

print("\n사이클 2: 구역 통계 추가")

# ── 사이클 2 ─────────────────────────────────────────────────

# 두 번째 Thought:

# - 사용자의 질문에 "창고 출입구 탐지 현황"도 들어 있으므로

# - 특정 구역의 탐지 수를 확인해야 한다.

#

# 두 번째 Action:

# - count_objects_in_zone Tool 실행

#

# 두 번째 Observation:

# - 창고 출입구의 총 탐지 수, 프레임 수, 평균 탐지 수

frames_json = json.dumps(frame_results, ensure_ascii=False)

obs2 = react_step(
    thought="창고 출입구의 탐지 빈도도 확인해야겠다.",
    action_name="count_objects_in_zone",
    action_input={"frames_json": frames_json, "zone": "창고 출입구"},
)

# ── Final Answer 생성 ────────────────────────────────────────

print("\n두 Observation 종합 → Final Answer 생성")

# obs1과 obs2는 JSON 문자열이므로 Python dict로 변환한다.

d_summary = json.loads(obs1)

d_zone = json.loads(obs2)

# 두 Tool의 결과를 종합해 최종 답변을 만든다.

#

# 실제 Agent에서는 이 Final Answer도 LLM이 자연어로 생성한다.

# 여기서는 코드로 직접 문자열을 조립한다.

final_answer = (
    f"분석 결과:\n"
    f" 전체: 정상 {d_summary['정상']}건, 주의 {d_summary['주의']}건, "
    f"위험 {d_summary['위험']}건\n"
    f" 위험 프레임 ID: {d_summary['위험_프레임_ids']}\n"
    f" 창고 출입구 탐지: 총 {d_zone['total_detections']}건 "
    f"({d_zone['frame_count']}개 프레임, 평균 {d_zone['avg_per_frame']}/프레임)"
)

print(f"\n✅ Final Answer:\n{final_answer}")
