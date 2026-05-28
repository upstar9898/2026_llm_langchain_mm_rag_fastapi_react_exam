# Whisper 설치 (openai 패키지와 다른 별도 패키지입니다)
# pip install openai-whisper

import os
import time
import whisper

# ─── 설정 ────────────────────────────────────────────────────
MODEL_NAME = "base"  # CPU: "base" / GPU: "turbo"
AUDIO_PATH = "./waves/20260526_All_units.wav"  # 변환할 음성 파일 경로

# ─────────────────────────────────────────────────────────────
# STEP 1: 모델 로딩
#
# 최초 실행 시 ~/.cache/whisper/ 에 모델 파일을 자동 다운로드합니다.
# 두 번째 실행부터는 캐시에서 바로 로드되므로 빠릅니다.
#
# download_root: 모델 저장 경로 지정 (기본: ~/.cache/whisper)
# device: "cpu" 또는 "cuda" (GPU)
#   → 지정 안 하면 GPU가 있을 때 자동으로 GPU 사용 -> 믿지 말 것
# ─────────────────────────────────────────────────────────────
print(f"모델 로딩 중: {MODEL_NAME}")
print("(최초 실행 시 모델 다운로드가 진행됩니다. 잠시 기다려주세요.)\n")

start = time.time()
model = whisper.load_model(
    MODEL_NAME,
    # download_root="./whisper_models",  # 모델 저장 위치를 직접 지정할 때
)
print(f"✅ 모델 로딩 완료: {time.time() - start:.1f}초\n")

# ─────────────────────────────────────────────────────────────
# STEP 2: 음성 파일 → 텍스트 변환 (transcribe)
#
# audio: 파일 경로(str), numpy 배열, torch.Tensor 모두 가능
#        WAV, MP3, MP4, M4A, FLAC 등 ffmpeg 지원 포맷 전부 가능
#
# language: 언어 코드 지정
#   지정하면 → 언어 감지 단계 생략 → 더 빠르고 정확
#   지정 안 하면 → 첫 30초 분석해서 자동 감지
#   한국어: "ko" / 영어: "en" / 일본어: "ja"
#
# task: "transcribe" (원어 그대로) / "translate" (영어로 번역)
#
# initial_prompt: 도메인 힌트 텍스트
#   Whisper에게 어떤 종류의 대화인지 미리 알려주면 인식 정확도가 올라감
#   예) "보안 무전 교신. 경비팀 교신 내용입니다."
#
# verbose: True → 실시간 세그먼트 출력
#          False → 조용히 실행
#          None → 아무것도 출력 안 함
#
# fp16: CPU 환경에서는 반드시 False
#       GPU 환경에서는 True로 바꾸면 처리 속도 약 2배 향상
# ─────────────────────────────────────────────────────────────

if not os.path.exists(AUDIO_PATH):
    print(f"⚠️ 오디오 파일이 없습니다: {AUDIO_PATH}")
    print("   WAV 또는 MP3 파일을 해당 경로에 넣고 다시 실행하세요.\n")
    print("   테스트용 무음 WAV 파일을 만들려면:")
    print("   ffmpeg -f lavfi -i anullsrc=r=16000:cl=mono -t 3 radio_sample.wav\n")
else:
    print(f"변환 시작 : {AUDIO_PATH}")

    start = time.time()

    result = model.transcribe(
        AUDIO_PATH,
        language="en",
        task="transcribe",  # translate로 지정하면 영어로 번역
        initial_prompt=(  # 음성에 대한 도메인 힌트 -> 정확도 향상
            "보안 무전 교신. 경비팀 교신 내용입니다. "
            "CCTV 탐지 보고 및 현장 대응 내용을 포함합니다."
        ),
        verbose=True,
        fp16=False,
    )

    elapsed = time.time() - start
    print(f"변환 완료: {elapsed:.1f}초\n")

    # ─────────────────────────────────────────────────────────
    # STEP 3: 결과 확인
    #
    # result 딕셔너리 구조:
    # {
    #   "text": "전체 변환 텍스트",
    #   "language": "ko",
    #   "segments": [
    #     {
    #       "start": 0.0,         ← 시작 시간 (초)
    #       "end": 3.2,           ← 종료 시간 (초)
    #       "text": "세그먼트 텍스트",
    #       "avg_logprob": -0.3,  ← 신뢰도 (0에 가까울수록 높음)
    #       "no_speech_prob": 0.1 ← 무음 확률 (낮을수록 음성이 명확)
    #     },
    #     ...
    #   ]
    # }
    # ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("📝 변환 결과")
    print("=" * 60)

    # 전체 텍스트
    print(f"\n[전체 텍스트]\n{result['text']}\n")

    # 감지된 언어
    print(f"[감지 언어] {result['language']}\n")

    # 세그먼트별 상세 결과
    # 세그먼트: Whisper가 문장 단위로 끊어서 반환하는 타임스탬프 정보
    print("[세그먼트별 타임스탬프]")
    for seg in result["segments"]:
        start_t = seg["start"]
        end_t = seg["end"]
        text = seg["text"].strip()
        prob = seg["avg_logprob"]  # 신뢰도 (낮을수록 불확실)
        no_sp = seg["no_speech_prob"]  # 무음 확률

        # avg_logprob 기준 신뢰도 표시
        # -0.5 이상  → 신뢰할 수 있는 인식 결과
        # -0.5 ~ -1.0 → 주의 필요
        # -1.0 이하  → 노이즈나 불명확한 발화일 가능성 높음
        confidence = "✅" if prob > -0.5 else ("⚠️" if prob > -1.0 else "❌")

        print(f"  {confidence} [{start_t:5.1f}s → {end_t:5.1f}s] {text}")
        print(f"       신뢰도: {prob:.3f} | 무음확률: {no_sp:.3f}")

    print()

    # ─────────────────────────────────────────────────────────
    # STEP 4: LangChain으로 텍스트 요약
    # 변환된 텍스트를 GPT-4o에게 넘겨 보안 보고 형식으로 요약
    # ─────────────────────────────────────────────────────────
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    load_dotenv()

    print("=" * 60)
    print("🤖 GPT-4o 무전 내용 요약")
    print("=" * 60)

    prompt = PromptTemplate.from_template(
        """당신은 보안 관제 요원입니다.
아래 무전 교신 내용을 보안 보고서 형식으로 요약하세요.

[무전 교신 원문]
{transcription}

다음 항목을 포함해서 요약하세요:
- 상황 요약 (1~2문장)
- 탐지 내용 (인원, 위치, 시간)
- 조치 사항
- 후속 대응 필요 여부"""
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm | StrOutputParser()

    summary = chain.invoke({"transcription": result["text"]})
    print(f"\n{summary}")
    print("\n✅ Whisper 로컬 변환 + LangChain 요약 완료!")
