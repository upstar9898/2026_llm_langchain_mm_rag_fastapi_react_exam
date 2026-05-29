"""
목소리(voice) · 모델(model) · 속도(speed) 비교 실습

핵심 질문: "경보용 목소리"와 "일반 안내용 목소리"는 어떻게 달라야 할까?
직접 들어보고 상황에 맞는 목소리를 선택하세요.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def compare_voices(text: str, save_dir: str = ".") -> None:
    """
    동일한 텍스트를 여러 목소리로 변환하여 파일로 저장합니다.
    각 파일을 재생해서 차이를 직접 들어보세요.

    목소리 선택 가이드 (CCTV 경보 시스템 기준):
    ┌──────────┬───────────────────────────────────────────┐
    │  voice   │  추천 상황                                │
    ├──────────┼───────────────────────────────────────────┤
    │ alloy    │ 시스템 일반 안내 (중립적)                  │
    │ echo     │ 차분한 정보 전달                           │
    │ fable    │ 내레이션·보고서 낭독                       │
    │ onyx     │ 긴급 경보 (낮고 권위적, 명확)             │
    │ nova     │ 주의 안내 (밝고 명확)                     │
    │ shimmer  │ 정상 상황 안내 (부드럽고 친근)            │
    └──────────┴───────────────────────────────────────────┘
    """
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    Path(save_dir).mkdir(exist_ok=True)
    print(f"텍스트: '{text[:50]}...'")
    print("-" * 55)

    for voice in voices:
        output_path = f"{save_dir}/voice_{voice}.mp3"

        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,  # ← 이 파라미터만 바뀝니다
            input=text,
            response_format="mp3",
        )
        # write_to_file() 이 가장 간단한 저장 방법입니다.
        response.write_to_file(output_path)

        file_size = Path(output_path).stat().st_size
        print(f"  [{voice:8s}] → {output_path} ({file_size:,} bytes)")

    print("\n💡 각 파일을 재생해서 상황에 맞는 목소리를 선택하세요!")


def compare_models(text: str, voice: str = "alloy") -> None:
    """
    tts-1 (빠름) vs tts-1-hd (고품질) 비교

    선택 기준:
    - 실시간 경보 → tts-1   (지연 최소화가 최우선)
    - 보고서 낭독 → tts-1-hd (음질이 최우선)
    """
    print("\n모델 비교:")
    print("-" * 55)

    model_info = {
        "tts-1": "빠름·저렴  → 실시간 경보 권장",
        "tts-1-hd": "고품질·비쌈 → 녹음·방송 권장",
    }

    for model, description in model_info.items():
        output_path = f"model_{model.replace('-', '_')}.mp3"
        response = client.audio.speech.create(
            model=model,  # ← 모델만 교체
            voice=voice,
            input=text,
            response_format="mp3",
        )
        response.write_to_file(output_path)
        file_size = Path(output_path).stat().st_size
        print(f"  [{model:10s}] {description} ({file_size:,} bytes)")


def compare_speeds(text: str) -> None:
    """
    speed 파라미터 비교 (범위: 0.25 ~ 4.0)

    상황별 추천 속도:
    - 긴급 경보   : 1.2  (약간 빠르게, 긴박감 표현)
    - 일반 안내   : 1.0  (기본값)
    - 외국인 대상 : 0.75 (천천히 또렷하게)

    주의: 너무 빠르면(2.0↑) 내용 전달 어려움
          너무 느리면(0.5↓) 어색하게 들림
    """
    print("\n속도 비교 (speed 파라미터):")
    print("-" * 55)

    speed_configs = [
        (0.75, "느림  — 외국인·고령자 대상 안내"),
        (1.0, "기본  — 일반 상황 안내"),
        (1.2, "빠름  — 긴급 경보 상황"),
    ]

    for speed, label in speed_configs:
        output_path = f"speed_{str(speed).replace('.', '_')}.mp3"
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            speed=speed,  # ← 속도 파라미터
            response_format="mp3",
        )
        response.write_to_file(output_path)
        file_size = Path(output_path).stat().st_size
        print(f"  [speed={speed:<4}] {label} ({file_size:,} bytes)")


if __name__ == "__main__":
    print("=" * 55)
    print("예제 2: 목소리 · 모델 · 속도 비교")
    print("=" * 55)

    test_text = (
        "CCTV 이상 상황 감지. 주차장 B구역에서 "
        "심야 시간대 3명이 배회 중입니다. 즉시 확인 바랍니다."
    )

    # compare_voices(test_text, save_dir="voices")
    compare_models(test_text)
    compare_speeds(test_text)
