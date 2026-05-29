# 이 예제에서는 이미지 파일 <--> base64 상호 변환하는 과정을 학습한다.

import base64
from pathlib import Path

# 확장자 → media_type 매핑 테이블
media_type_map = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}


def image_to_base64(image_path: str) -> tuple[str, str]:
    """
    이미지 파일을 Base64 문자열로 변환한다.

    왜 media_type도 같이 반환하는가?
        API에 이미지를 보낼 때 "이것은 JPEG입니다", "이것은 PNG입니다" 라고
        형식을 함께 알려줘야 한다. 이것이 media_type(= MIME 타입)이다.

    Returns:
        (base64_문자열, media_type)  예: ("/9j/4AA...", "image/jpeg")
    """
    ext = image_path.split(".")[-1].lower()  # 확장자 추출

    media_type = media_type_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:  # "rb" = read binary
        raw_bytes = f.read()  # 파일 전체 읽기
        b64_string = base64.b64encode(raw_bytes).decode("utf-8")  # byte -> str

    return b64_string, media_type


def base64_to_image(b64: str, save_path: str) -> None:
    image_bytes = base64.b64decode(b64)
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 폴더 없으면 생성
    out_path.write_bytes(image_bytes)  # bytes -> file write


b64, mt = image_to_base64("./vision_sample/cat.jpeg")

print(f"media_type : {mt}")
print(f"Base64 길이: {len(b64)} 문자")
print(f"앞 30자    : {b64[:30]}...")
print(f"data URI   : data:{mt};base64,{b64[:20]}...")

# b64에 들어있는 base64를 디코딩하여 "./vision_sample/decode_exam.jpeg"로 저장하는 함수를 만드세요
# base64_to_image(b64, file_path)

base64_to_image(b64, "./vision_sample/decode_exam.jpeg")
