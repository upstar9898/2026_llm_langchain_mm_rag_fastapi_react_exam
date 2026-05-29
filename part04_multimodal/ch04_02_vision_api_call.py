import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import base64
from ch04_01_visionModel_basic import image_to_base64

## 일반 API vs Vision API 메시지 구조 비교

# 일반 텍스트 API는 `content` 에 문자열 하나만 넣는다.
# Vision API는 **이미지 블록과 텍스트 블록을 리스트로** 묶어서 넣는다.

# 일반 텍스트 API
messages = [
    {"role": "user", "content": "파이썬이 뭔가요?"}  # 문자열 하나
]

# Vision API
messages = [
    {
        "role": "user",
        "content": [  # 리스트!
            {"type": "image_url", "image_url": {"url": "..."}},  # 이미지 블록
            {"type": "text", "text": "이 이미지를 설명해주세요."},  # 텍스트 블록
        ],
    }
]


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_image(image_path: str, prompt: str) -> dict:
    """
    이미지 1장을 Vision API로 분석하고 결과를 dict로 반환한다.

    Args:
        image_path : 분석할 이미지 파일 경로
        prompt     : 분석 지시 (예: "이 이미지의 주제를 JSON으로 알려주세요.")
    Returns:
        LLM이 반환한 JSON을 파싱한 dict
    """
    b64, mt = image_to_base64(image_path=image_path)

    # vision model 호출
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [  # 리스트!
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mt};base64,{b64}",
                            # detail 옵션:
                            #   "low"  → 빠름, 저렴, 간단한 분류에 적합
                            #   "high" → 느림, 비쌈, 세부 내용 분석에 적합
                            "detail": "low",
                        },
                    },  # 이미지 블록
                    {
                        "type": "text",
                        "text": prompt,
                    },  # 텍스트 블록
                ],
            }
        ],
    )

    content = response.choices[0].message.content
    return json_parse(content)


def json_parse(text: str) -> dict:
    """
    LLM 응답에서 JSON을 안전하게 파싱한다.

    왜 이 함수가 필요한가?
        Vision API는 response_format 파라미터를 지원하지 않는다.
        그래서 LLM이 가끔 아래처럼 코드 펜스로 감싸서 반환한다:
            ```json
            {"subject": "고양이", ...}
            ```
        이 함수는 그 코드 펜스를 자동으로 제거하고 dict로 변환한다.
    """
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())  # 앞쪽 ``` 제거
    text = re.sub(r"\s*```$", "", text)  # 뒤쪽 ``` 제거
    return json.loads(text.strip())


if __name__ == "__main__":
    # ── 실행 ──────────────────────────────────────────────────────
    prompt = """이 이미지를 분석해서 아래 JSON 형식으로만 응답하세요.
    {
    "subject": "이미지의 주요 피사체",
    "description": "전체적인 설명 (2~3문장)",
    "mood": "이미지의 분위기",
    "colors": ["주요 색상1", "주요 색상2"]
    }"""

    result = analyze_image("./vision_sample/ichika.webp", prompt)
    print(json.dumps(result, ensure_ascii=False, indent=2))
