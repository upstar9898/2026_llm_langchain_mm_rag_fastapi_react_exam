from dotenv import load_dotenv

from whisper_transcriber import (
    transcribe_audio,
    print_transcription_result,
)

from summary_topic_extractor import extract_topic, extract_summary

from rag_store import (
    create_transcript_document,
    add_transcript_to_vectorstore
)


# ─────────────────────────────────────────────────────────────
# 환경변수 로드
# ─────────────────────────────────────────────────────────────
load_dotenv()


# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
AUDIO_PATH = "./audio/20260526_“All_units (1).wav"

WHISPER_MODEL = "base"

CHROMA_DIR = "./chroma_db"

COLLECTION_NAME = "whisper_transcripts"


# ─────────────────────────────────────────────────────────────
# STEP 1: Whisper 전사
# ─────────────────────────────────────────────────────────────
transcript = transcribe_audio(
    audio_path=AUDIO_PATH,
    model_name=WHISPER_MODEL,
    language="en",
    task="transcribe",
    initial_prompt=(
        "보안 무전 교신. "
        "경비팀 교신 내용입니다. "
        "CCTV 탐지 보고 및 현장 대응 내용을 포함합니다."
    ),
    fp16=False,
)

print_transcription_result(transcript)


# ─────────────────────────────────────────────────────────────
# STEP 2: LLM으로 대화 주제 추출
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("🧠 대화 주제 추출")
print("=" * 60)

topic = extract_topic(transcript["text"])
summary = extract_summary(transcript["text"])
print(f"주제: {topic}\n")
print(f"요약: {summary}\n")

# ─────────────────────────────────────────────────────────────
# STEP 3: Document 생성
# ─────────────────────────────────────────────────────────────
document = create_transcript_document(
    transcript=transcript,
    topic=topic,
    summary=summary,
)

print("=" * 60)
print("📄 생성된 Document")
print("=" * 60)

print(document.page_content[:500])
print()
print(document.metadata)
print()


# ─────────────────────────────────────────────────────────────
# STEP 4: ChromaDB에 추가 저장
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("💾 ChromaDB 추가 저장")
print("=" * 60)

vectorstore = add_transcript_to_vectorstore(
    transcript=transcript,
    topic=topic,
    summary=summary,
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME,
)

print("✅ 추가 저장 완료\n")