# 한 번의 API 호출로 이미지 여러 장을 분석시키기

import json, os, re, base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from ch04_01_visionModel_basic import image_to_base64
from ch04_02_vision_api_call import json_parse


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_multiple_images(image_paths: list[str], prompt: str) -> dict:
    content_blocks = []
    for path in image_paths:
        b64, mt = image_to_base64(image_path=path)
        content_blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mt};base64,{b64}",
                    "detail": "low",
                },
            }
        )

    content_blocks.append(
        {
            "type": "text",
            "text": prompt,
        },
    )
    # vision model 호출

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": content_blocks,
            }
        ],
    )

    content_blocks = response.choices[0].message.content
    return json_parse(content_blocks)


# ── 실행 ──────────────────────────────────────────────────────
image_files = [
    "./vision_sample/cat.jpeg",
    "./vision_sample/hamster.jpeg",
    "./vision_sample/blue_parrot.jpeg",
]


prompt = """위 이미지들을 순서대로 분석해서 아래 JSON 형식으로만 응답하세요.
{
  "images": [
    {"index": 1, "subject": "피사체", "description": "설명"},
    {"index": 2, "subject": "피사체", "description": "설명"},
    {"index": 3, "subject": "피사체", "description": "설명"}
  ],
  "common_theme": "세 이미지의 공통 주제"
}"""

print("이미지 로딩 중...")
result = analyze_multiple_images(image_files, prompt)
print("\n=== 분석 결과 ===")
print(json.dumps(result, ensure_ascii=False, indent=2))
