import cv2
import easyocr
import re
import threading
import time

# EasyOCR 초기화
reader = easyocr.Reader(['ko', 'en'], gpu=True)

# 카메라 열기
cap = cv2.VideoCapture(0)

# OCR 결과 저장용 전역 변수
ocr_results = []
ocr_lock = threading.Lock()

# OCR 수행 간격 (초)
ocr_interval = 0.5
last_ocr_time = 0

# OCR을 쓰레드에서 실행하는 함수
def run_ocr(frame):
    global ocr_results
    results = reader.readtext(frame)
    detected_numbers = []
    for bbox, text, conf in results:
        match = re.findall(r"\d{3}\s?[A-Z]\d{2}", text)
        if match:
            detected_numbers.extend(match)
    with ocr_lock:
        ocr_results = [(results, detected_numbers)]

# 첫 프레임 OCR 즉시 실행
ret, first_frame = cap.read()
if ret:
    threading.Thread(target=run_ocr, args=(first_frame.copy(),), daemon=True).start()
    last_ocr_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임을 가져올 수 없습니다.")
        break

    current_time = time.time()
    # 일정 간격마다 OCR 쓰레드 실행
    if current_time - last_ocr_time > ocr_interval:
        threading.Thread(target=run_ocr, args=(frame.copy(),), daemon=True).start()
        last_ocr_time = current_time

    # OCR 결과 가져오기
    with ocr_lock:
        results_copy = ocr_results.copy()

    # 화면에 OCR 결과 표시
    if results_copy:
        results, detected_numbers = results_copy[0]
        for bbox, text, conf in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if detected_numbers:
            print("운송장 번호 후보:", detected_numbers)

    cv2.imshow("Waybill OCR", frame)

    # 종료 키
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
