from pygoogle_image import image as pi
import os

search_term = input("검색할 이미지 키워드를 입력하세요: ")

# 다운로더 호출
try:
    # limit으로 다운로드할 이미지 개수 조절 가능
    pi.download(search_term, limit=100)
    print(f"\n'{search_term}' 이미지 다운로드가 완료되었습니다.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")