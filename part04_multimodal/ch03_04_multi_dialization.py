import os
import json
import re
import soundfile as sf
import torch
from dotenv import load_dotenv

import whisper
from pyannote.audio import Pipeline

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ───────────────── 설정 ─────────────────
MODEL_NAME = "base"
AUDIO_PATH = "./waves/20260526_All_units.wav"  # 변환할 음성 파일 경로
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

SPEAKER_NAMES = {
    "SPEAKER_00": "민욱 (Control)",
    "SPEAKER_01": "주완 (Unit 3)",
    "SPEAKER_02": "대진 (Unit 2)",
    "SPEAKER_03": "다은 (Unit 5)",
}


def correct_timestamps(segments: list, actual_duration: float) -> tuple:
    """
    Whisper 타임스탬프 보정 함수.

    Windows에서 ffmpeg 없이 실행하면 Whisper가 오디오를
    16000Hz로 리샘플링하지 못해서 타임스탬프가 뻥튀기됩니다.
    예) 44100Hz WAV → 타임스탬프가 실제의 2.75배 (44100/16000)

    실제 오디오 길이 기준으로 전체 비율을 맞춰 보정합니다.
    실제 길이와 1초 이내 차이면 보정 없이 원본 반환합니다.

    반환값: (보정된 segments, 보정 비율)
    pyannote annotation 보정에도 동일 비율을 사용합니다.
    """
    if not segments:
        return segments, 1.0

    whisper_max = segments[-1]["end"]  # 세그먼트의 끝 구간

    if abs(whisper_max - actual_duration) < 1.0:
        return segments, 1.0

    ratio = actual_duration / whisper_max
    print(
        f"  ⚠️ 타임스탬프 보정: {whisper_max:.1f}s → {actual_duration:.1f}s (비율: {ratio:.4f})"
    )

    corrected = []
    for seg in segments:
        s = seg.copy()
        s["start"] = round(seg["start"] * ratio, 2)
        s["end"] = round(seg["end"] * ratio, 2)
        corrected.append(s)

    return corrected, ratio


def get_speaker_at(annotation, start: float, end: float) -> str:
    """
    Whisper 세그먼트의 중간 시점(mid)이 속하는 화자를 반환합니다.

    중간 시점을 쓰는 이유:
    Whisper 세그먼트와 pyannote 구간의 경계가 정확히 일치하지 않습니다.
    중간값을 쓰면 경계 오류를 줄일 수 있습니다.

    끝점을 열린 구간(<)으로 처리하는 이유:
    [0.0, 4.5), [4.5, 8.2) 처럼 경계에서 맞닿을 때
    mid == 4.5 이면 두 구간 모두 매칭될 수 있습니다.
    < 로 처리하면 중복 매핑을 방지합니다.
    """

    # 화자 이름 매핑
    # pyannote는 SPEAKER_00, SPEAKER_01 ... 으로 반환합니다.
    # 코드 실행 후 "감지된 화자 구간" 출력을 보고 순서에 맞게 조정하세요.

    mid = (start + end) / 2
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        if turn.start <= mid < turn.end:
            return speaker
    return "UNKNOWN"


def merge_consecutive_segment(segments: list) -> list:
    """
    연속된 같은 화자의 발화를 하나로 합칩니다.

    예시:
    [민욱] "Subject attempted access."  7.8s ~ 10.0s
    [민욱] "Access denied three times." 10.1s ~ 12.0s
    →
    [민욱] "Subject attempted access. Access denied three times." 7.8s ~ 12.0s

    gap_threshold: 0.8초 이내 같은 화자 → 연속 발화로 판단
    """
    """
    "speaker": speaker_id,
            "speaker_name": speaker_name,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
    """

    gap_threshold = 0.8
    merged = [segments[0].copy()]

    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        if seg["speaker"] == prev["speaker"] and gap < gap_threshold:
            prev["text"] += "  " + seg["text"].strip()
            prev["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    return merged


# ─────────────────────────────────────────────────────────────
# STEP 1: Whisper STT
#
# language="en": 영어 무전 → "ko"로 바꾸면 한국어
# fp16=False:    CPU 환경 필수. GPU는 True로 변경 시 2배 빠름
# initial_prompt: 도메인 힌트 → 인식 정확도 향상
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Whisper STT")
print("=" * 60)

whisper_model = whisper.load_model(MODEL_NAME)

result = whisper_model.transcribe(
    AUDIO_PATH,
    language="en",
    task="transcribe",
    initial_prompt=(
        "Security radio communication. "
        "Units responding to suspicious activity at parking structure."
    ),
    verbose=False,
    fp16=False,
)

print(f"✅ STT 완료 — 세그먼트 수: {len(result['segments'])}개")
print(f"   전체 텍스트 미리보기: {result['text'][:80]}...\n")

# 타임스탬프 보정
# soundfile을 이용하여 실제 오디오 길이를 가져와서 보정 비율 계산
# Whisper 세그먼트와 pyannote 세그먼트 구간에 동일 비율 적용
actual_duration = sf.info(AUDIO_PATH).duration
print(f"실제 오디오 길이 : {actual_duration}초")


# ─────────────────────────────────────────────────────────────
# STEP 2: pyannote 화자 분리
#
# Pipeline.from_pretrained: 모델 최초 로드 시 HF에서 자동 다운로드
# 두 번째 실행부터는 캐시에서 바로 로드 (~/.cache/huggingface/)
#
# soundfile.read(): torchcodec / FFmpeg 없이 WAV 읽기
#   → Windows에서 FFmpeg full-shared 미설치 문제 완전 우회
#   → torchaudio.load() 대신 사용
#
# waveform.unsqueeze(0): (time,) → (1, time) 채널 차원 추가
#   pyannote 입력 형식: {"waveform": (channel, time) Tensor, "sample_rate": int}
#
# min_speakers / max_speakers: num_speakers 고정 대신 범위 지정
#   → 오디오가 짧거나 화자가 적을 때 경고 없이 유연하게 처리
#
# pyannote 4.x 주의:
#   반환값이 DiarizeOutput 객체 (pyannote 3.x의 Annotation과 다름)
#   .speaker_diarization 필드로 Annotation 객체 꺼내서 사용
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: pyannote 화자 분리")
print("=" * 60)

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN,  # use_auth_token 아님! pyannote 4.x는 token=
)

# wav 읽기 -> soundfile 라이브러리 사용 (ffmpeg 불필요)
data, sample_rate = sf.read(AUDIO_PATH, dtype="float32")
# (time, ) -> (1, time) "1은 MONO"
waveform = torch.from_numpy(data).unsqueeze(0)


output = diarization_pipeline(
    {"waveform": waveform, "sample_rate": sample_rate},
    min_speakers=2,  # 최소 화자 수 (num_speakers 고정 대신 범위 지정)
    max_speakers=4,  # 최대 화자 수
)

annotation = output.speaker_diarization

print("✅ 화자 분리 완료")
print("   감지된 화자 구간:")
for turn, _, speaker in annotation.itertracks(yield_label=True):
    print(f"   {speaker}: {turn.start:.1f}s ~ {turn.end:.1f}s")
print()

# ─────────────────────────────────────────────────────────────
# STEP 3: STT + 화자 병합
#
# 각 Whisper 세그먼트의 중간 시점이
# pyannote 어떤 화자 구간에 속하는지 확인해서 화자를 붙입니다.
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: STT + 화자 병합")
print("=" * 60)

segments_with_speaker = []
for seg in result["segments"]:
    speaker_id = get_speaker_at(annotation, seg["start"], seg["end"])
    speaker_name = SPEAKER_NAMES.get(speaker_id, speaker_id)
    segments_with_speaker.append(
        {
            "speaker": speaker_id,
            "speaker_name": speaker_name,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        }
    )

merged_segments = merge_consecutive_segment(segments_with_speaker)
print(
    f"✅ 병합 완료 — {len(result['segments'])}개 세그먼트 → {len(merged_segments)}개\n"
)


# ─────────────────────────────────────────────────────────────
# STEP 4: 화자별 대화 전사 출력
#
# 전사(Transcript): 음성을 텍스트로 옮긴 것
# 화자 이름 + 시간 정보 + 발화 내용을 보기 좋게 출력합니다.
# 전사 결과를 텍스트 파일로도 저장합니다.
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 4: 화자별 대화 전사")
print("=" * 60)
print()

for seg in merged_segments:
    # 초 단위 → 분:초 형식으로 변환
    start_min = int(seg["start"]) // 60
    start_sec = seg["start"] % 60
    end_min = int(seg["end"]) // 60
    end_sec = seg["end"] % 60

    print(
        f"┌─ [{seg['speaker_name']}]  "
        f"{start_min:02d}:{start_sec:05.2f} → {end_min:02d}:{end_sec:05.2f}"
    )
    print(f"│  {seg['text']}")
    print("│")

print("└─ (전사 종료)")
print()

# 전사 결과를 텍스트 파일로 저장
transcript_path = AUDIO_PATH.replace(".wav", "_transcript.txt")
with open(transcript_path, "w", encoding="utf-8") as f:
    f.write("=== 화자별 대화 전사 ===\n\n")
    for seg in merged_segments:
        f.write(f"[{seg['speaker_name']}] ({seg['start']:.1f}s ~ {seg['end']:.1f}s)\n")
        f.write(f"{seg['text']}\n\n")

print(f"📄 전사 파일 저장: {transcript_path}\n")
