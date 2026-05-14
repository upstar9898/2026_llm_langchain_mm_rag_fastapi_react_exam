# ─────────────────────────────────────────────────────────────
# 예제 2: 대화 이력 저장소 직접 구현
#
# ConversationBufferMemory / InMemoryChatMessageHistory가
# 내부적으로 어떻게 동작하는지 이해하기 위한 교육용 코드입니다.
# 실제로는 LangChain이 제공하는 클래스를 사용합니다.
# ─────────────────────────────────────────────────────────────


class SimpleBufferMemory:
    """
    대화 이력을 리스트에 저장하는 간단한 메모리 클래스.

    LangChain의 InMemoryChatMessageHistory와 동일한 역할을 합니다.
    직접 구현해 보면서 Memory의 원리를 이해합니다.
    """

    def __init__(self):
        # 대화 이력을 저장하는 리스트
        # 각 항목: {"role": "human" 또는 "ai", "content": "내용"}
        self.messages: list = []

    def add_user_message(self, text: str):
        """사용자(운영자)의 메시지를 이력에 추가"""
        self.messages.append({"role": "human", "content": text})

    def add_ai_message(self, text: str):
        """AI의 응답을 이력에 추가"""
        self.messages.append({"role": "ai", "content": text})

    def get_all_messages(self) -> list:
        """저장된 전체 대화 이력 반환 — LLM 호출 시 사용"""
        return self.messages

    def format_as_text(self) -> str:
        """
        대화 이력을 사람이 읽기 좋은 텍스트로 변환.

        프롬프트에 직접 삽입할 때 이 형태를 사용합니다.

        출력 형태:
            Human: ...
            AI: ...
            Human: ...
        """
        lines = []
        for msg in self.messages:
            prefix = "Human" if msg["role"] == "human" else "AI"
            lines.append(f"{prefix}: {msg['content']}")
        return "\\n".join(lines)

    def clear(self):
        """대화 이력 초기화 — 새 운영자 세션 시작 시 사용"""
        self.messages = []

    def __len__(self):
        """저장된 메시지 수 반환 (len(memory) 형태로 사용)"""
        return len(self.messages)


# ── 실제 사용 시나리오 ──────────────────────────────────────
memory = SimpleBufferMemory()

# 프레임 분석 결과를 하나씩 메모리에 저장
memory.add_user_message("1번 프레임 분석해줘. person 1명, 13:25, 주차장")
memory.add_ai_message('{"frame_id": 1, "risk_level": "정상", "reason": "주간 정상 보행자"}')

memory.add_user_message("3번 프레임 분석해줘. person 2명, 02:13, 창고 출입구")
memory.add_ai_message('{"frame_id": 3, "risk_level": "위험", "reason": "심야 창고 2인 배회"}')

# 후속 질문 추가
memory.add_user_message("3번 프레임 왜 위험으로 분류했어?")

# 저장된 이력 확인
print("[현재 메모리에 저장된 대화 이력]")
print(memory.format_as_text())
print(f"\\n총 {len(memory)}개 메시지 저장됨")

# LLM에 전달할 형태로 가져오기
all_msgs = memory.get_all_messages()
print(f"\\nLLM에 전달될 메시지 수: {len(all_msgs)}개")
print("마지막 메시지:", all_msgs[-1]["content"])