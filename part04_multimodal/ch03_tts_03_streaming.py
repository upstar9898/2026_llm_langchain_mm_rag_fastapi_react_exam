"""
긴 텍스트 청크 분할 + stream_to_file() 스트리밍 저장

왜 필요한가?
  [일반 방식]
    전체 텍스트 한 번에 요청 → API 처리(10초) 대기 → 재생 시작
    → 사용자가 10초를 기다려야 함

  [청크 분할 + 스트리밍]
    청크1만 먼저 요청(2초) → 즉시 재생 시작
    → 재생하는 동안 청크2, 3 순차 변환
    → 사용자는 2초만 기다리면 됨!

  경보 시스템에서는 "첫 마디를 얼마나 빨리 들을 수 있는가"가 핵심입니다.
"""

import re
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def split_into_sentences(text: str, max_chars: int = 200) -> list:
    """
    긴 텍스트를 문장 단위로 분할합니다.

    분할 기준:
    1. 마침표(.), 느낌표(!), 물음표(?) 뒤에서 분할
    2. max_chars 이하로 문장들을 합칩니다 (API 호출 횟수 최소화)
    3. 단일 문장이 max_chars를 초과하면 그대로 1개 청크로 처리

    매개변수:
        text      : 분할할 원본 텍스트
        max_chars : 청크 최대 길이 (기본 200자)

    반환값:
        청크 문자열 리스트

    사용 예시:
        chunks = split_into_sentences("긴 텍스트...", max_chars=200)
        # → ["첫 번째 청크.", "두 번째 청크."]
    """
    sentences = re.split(r"(?<=[.!?\n])\s*", text.strip())

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # max_chars 이하로 문장들을 합칩니다 (API 호출 횟수 최소화)
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            # 꽉 찼으면 저장하고, 새 청크 시작
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def tts_streaming_save(
    text: str, output_dir: str = "briefing_chunks", voice: str = "onyx"
) -> list:
    """
    긴 텍스트를 청크로 나눠 순차 변환하고
    stream_to_file()로 스트리밍 저장합니다.

    stream_to_file() vs write_to_file():
    - write_to_file() : 전체를 메모리에 올린 뒤 한 번에 씀
    - stream_to_file(): 네트워크에서 받는 즉시 조금씩 씀
                        → 대용량 파일, 메모리 절약 시 유리

    반환값: 저장된 MP3 파일 경로 리스트 (순서대로)
    """
    Path(output_dir).mkdir(exist_ok=True)
    chunks = split_into_sentences(text)

    print(f"총 {len(chunks)}개 청크로 분할:")

    for i, chunk in enumerate(chunks, 1):
        print(f"청크 {i:02d} ({len(chunk):3d}자): {chunk[:45]}...")

    print(f"스트리밍 변환 저장 시작 voice={voice}")

    saved_files = []
    for i, chunk in enumerate(chunks, 1):
        output_path = f"{output_dir}/chunk_{i:02d}.mp3"

        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=voice,
            input=chunk,
            response_format="mp3",
        ) as response:
            response.stream_to_file(output_path)

        if Path(output_path).stat().st_size > 0:
            saved_files.append(output_path)

        print(f"[청크 {i:02d}] → {output_path}")

    return saved_files


if __name__ == "__main__":
    print("=" * 55)
    print("예제 5: 보안 브리핑 긴 텍스트 스트리밍 변환")
    print("=" * 55 + "\n")

    # 보안 브리핑 텍스트 (긴 텍스트 청크 분할 시나리오)
    security_briefing = (
        "오늘 새벽 2시부터 4시 사이에 총 3건의 이상 상황이 감지되었습니다. "
        "첫 번째 사건은 주차장 A구역에서 발생했습니다. "
        "새벽 2시 13분, 2명의 인물과 차량 1대가 탐지되었으며 위험도는 위험으로 판정되었습니다. "
        "두 번째 사건은 창고 출입구에서 발생했습니다. "
        "새벽 3시 5분, 1명의 인물이 반복적으로 출입문을 조작하는 행동이 감지되었습니다. "
        "세 번째 사건은 공장 외곽 펜스 근처에서 발생했습니다. "
        "새벽 3시 47분, 3명이 집결하여 배회하는 상황이 확인되었습니다. "
        "현재 경비팀이 모두 출동하였으며, 경찰에도 상황을 전파하였습니다. "
        "지속적인 모니터링을 부탁드립니다."
    )

    print(f"원본 텍스트: {len(security_briefing)}자\n")

    files = tts_streaming_save(security_briefing, output_dir := "briefing_chunks")

    print(f"✅ 총 {len(files)}개 청크 파일 생성 완료")
    print("💡 실제 사용 시:")
    print(" - 파일을 순서대로 재생하면 전체 브리핑을 들을 수 있습니다.")
    print(" - 첫 번째 청크가 완성되는 즉시 재생하면 대기 시간이 줄어듭니다.")
