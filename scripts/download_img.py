import os
import cv2
import pytesseract
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# ocr 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

SEARCH_TERMS = [
    "쿠팡 운송장", "쿠팡 송장", "쿠팡 배송 라벨", "쿠팡 택배 상자",
    "coupang waybill", "coupang shipping label", "korean shipping label",
    "택배 송장", "운송장 사진", "택배 상자 스티커", "delivery barcode label"
]
SAVE_DIR = "dataset_coupang"
RAW_DIR = os.path.join(SAVE_DIR, "raw")
FILTERED_DIR = os.path.join(SAVE_DIR, "filtered")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(FILTERED_DIR, exist_ok=True)

def is_waybill_like(img):
    """운송장처럼 흰/파랑 배경 + 바코드 + 네모 구조인지 체크"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect_count = 0
    barcode_like = False

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h)

        if 2 < aspect < 6 and w > 100 and h > 20:
            rect_count += 1

        roi = gray[y:y+h, x:x+w]
        if w > 80 and h > 50:
            vertical_lines = cv2.Sobel(roi, cv2.CV_8U, 1, 0, ksize=3)
            if np.mean(vertical_lines) > 40:
                barcode_like = True

    return rect_count >= 1 and barcode_like

def has_text(img):
    """OCR을 통해 숫자/문자가 있는지 여부 확인"""
    text = pytesseract.image_to_string(img, lang="eng+kor")
    return len(text.strip()) > 5

def download_image(url, filename):
    try:
        r = requests.get(url, timeout=5)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img_np = np.array(img)[:, :, ::-1]
        cv2.imwrite(filename, img_np)
        return img_np
    except:
        return None

def search_and_filter():
    idx = 0
    for term in SEARCH_TERMS:
        print(f"[*] Searching: {term}")
        url = f"https://www.google.com/search?q={quote(term)}&tbm=isch"

        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        html = requests.get(url, headers=headers).text
        soup = BeautifulSoup(html, "html.parser")

        images = soup.find_all("img")
        for img_tag in tqdm(images):
            src = img_tag.get("src")
            if not src or "http" not in src:
                continue

            raw_path = os.path.join(RAW_DIR, f"{idx}.jpg")
            img_np = download_image(src, raw_path)
            if img_np is None:
                continue

            if is_waybill_like(img_np) or has_text(img_np):
                cv2.imwrite(os.path.join(FILTERED_DIR, f"{idx}.jpg"), img_np)

            idx += 1

if __name__ == "__main__":
    print("이미지 검색 / 다운로드")
    search_and_filter()
    print("다운로드 완료")
