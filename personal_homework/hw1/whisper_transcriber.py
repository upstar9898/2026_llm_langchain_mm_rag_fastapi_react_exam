import os
import time
from typing import Any, Dict

import whisper

_LOADED_MODELS = {}

def load_whisper_model(
    model_name: str = "base",
    device: str | None = None,
):
    # model_name + device 조합으로 캐시 키 생성
    cache_key = f"{model_name}:{device}"

    # 이미 로드된 모델이 있으면 재사용
    if cache_key in _LOADED_MODELS:
        print(f"♻️ 기존 Whisper 모델 재사용: {model_name}")
        return _LOADED_MODELS[cache_key]

    print(f"모델 로딩 중: {model_name}")

    start = time.time()

    if device is None:
        model = whisper.load_model(model_name)
    else:
        model = whisper.load_model(
            model_name,
            device=device,
        )

    elapsed = time.time() - start

    print(f"✅ 모델 로딩 완료: {elapsed:.1f}초\n")

    # 캐시에 저장
    _LOADED_MODELS[cache_key] = model

    return model


def transcribe_audio(
    audio_path: str,
    model_name: str = "base",
    language: str | None = None,
    task: str = "transcribe",
    initial_prompt: str | None = None,
    device: str | None = None,
    fp16: bool = False,
    verbose: bool | None = True,
) -> Dict[str, Any]:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"오디오 파일이 없습니다: {audio_path}")

    model = load_whisper_model(model_name=model_name, device=device)

    print(f"변환 시작: {audio_path}")
    start = time.time()

    result = model.transcribe(
        audio_path,
        language=language,
        task=task,
        initial_prompt=initial_prompt,
        verbose=verbose,
        fp16=fp16,
    )

    elapsed = time.time() - start
    print(f"✅ 변환 완료: {elapsed:.1f}초\n")

    segments = result.get("segments", [])

    duration = 0.0
    if segments:
        duration = float(segments[-1].get("end", 0.0))

    return {
        "source": os.path.basename(audio_path),
        "audio_path": audio_path,
        "text": result.get("text", "").strip(),
        "language": result.get("language", language),
        "duration": duration,
        "segments": segments,
        "elapsed": elapsed,
    }


def print_transcription_result(transcript: Dict[str, Any]) -> None:
    print("=" * 60)
    print("📝 변환 결과")
    print("=" * 60)

    print(f"\n[파일] {transcript['source']}")
    print(f"[감지 언어] {transcript['language']}")
    print(f"[길이] {transcript['duration']:.1f}초")

    print(f"\n[전체 텍스트]\n{transcript['text']}\n")

    print("[세그먼트별 타임스탬프]")

    for seg in transcript["segments"]:
        start_t = seg.get("start", 0.0)
        end_t = seg.get("end", 0.0)
        text = seg.get("text", "").strip()
        prob = seg.get("avg_logprob", 0.0)
        no_sp = seg.get("no_speech_prob", 0.0)

        confidence = "✅" if prob > -0.5 else ("⚠️" if prob > -1.0 else "❌")

        print(f"  {confidence} [{start_t:5.1f}s → {end_t:5.1f}s] {text}")
        print(f"       신뢰도: {prob:.3f} | 무음확률: {no_sp:.3f}")

    print()
