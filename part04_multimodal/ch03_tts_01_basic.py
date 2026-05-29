"""
TTS(Text-to-Speech) 기본 변환 예제입니다.

이 예제의 목표:

1. .env 파일에서 OpenAI API 키를 읽어옵니다.

2. OpenAI TTS API를 호출합니다.

3. 입력한 텍스트를 MP3 음성 파일로 저장합니다.

실행 전 준비:

pip install openai python-dotenv

.env 파일 내용:

OPENAI_API_KEY=sk-...실제_API_키...

실행:

python step04_ch03_tts_01_basic.py
"""

from pathlib import Path

from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

client = OpenAI()


def text_to_speech(
    text: str,
    output_path: str,
    voice: str = "alloy",
    model: str = "tts-1",
    speed: float = 1.0,
) -> bool:
    """
    텍스트를 음성으로 변환하여 WAV 파일로 저장합니다.

    반환값: True(성공) / False(실패)

    사용 예시:
        text_to_speech("안녕하세요", "hello.wav", voice="nova")
    """
    try:
        response = client.audio.speech.create(
            model=model, voice=voice, input=text, response_format="wav", speed=speed
        )

        # 응답 데이터를 파일로 직접 저장
        response.write_to_file(output_path)

        # 저장한 파일 크기 확인
        file_size = Path(output_path).stat().st_size

        print(f"  ✅ 저장 완료: {output_path} ({file_size:,} bytes)")

    except Exception as e:
        print(e)

    return False


if __name__ == "__main__":
    print("=" * 55)
    print("예제 1: TTS 기본 변환")
    print("=" * 55)

    # CCTV 경보 상황을 가정한 예제 문장입니다.
    # 실제 프로젝트에서는 LLM이 생성한 경보 문장이 이 자리에 들어갈 수 있습니다.

    alert_text = (
        "주의! 주차장 A구역에서 이상 상황이 감지되었습니다. "
        "새벽 2시 13분, 2명의 인물이 탐지되었으며 "
        "위험도는 주의 수준입니다. 경비원 확인을 요청합니다."
    )

    print(f"변환할 텍스트 ({len(alert_text)}자):")
    print(alert_text)

    text_to_speech(
        text=alert_text,
        output_path="cctv_alert_basic.wav",
        voice="alloy",
        model="tts-1",
        speed=1.0,
    )
