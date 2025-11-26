import cv2
import numpy as np
import os

# ======================
# 템플릿 로딩
# ======================
TEMPLATE_DIR = "templates"   # 100.png ~ 600.png 저장
TEMPLATES = {}

def load_templates():
    global TEMPLATES
    nums = [100,200,300,400,500,600]
    for num in nums:
        path = os.path.join(TEMPLATE_DIR, f"{num}.png")

        t = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if t is None:
            print("[ERR] Missing template:", path)
            continue

        t = cv2.resize(t, (200,100))
        _, t = cv2.threshold(t, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        TEMPLATES[num] = t

    print("[INFO] Loaded templates:", list(TEMPLATES.keys()))
    return len(TEMPLATES) > 0


# ======================
# 숫자 인식
# ======================
def recognize_unload_by_template(roi):
    if roi is None or roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (200,100))
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    best_num = None
    best_score = -1

    for num, templ in TEMPLATES.items():
        res = cv2.matchTemplate(th, templ, cv2.TM_CCOEFF_NORMED)
        score = float(res.max())

        if score > best_score:
            best_score = score
            best_num = num

    if best_score >= 0.65:
        return best_num
    return None


# ======================
# 메인 호출 함수
# ======================
def detect_unload_number(frame, debug=False):
    h, w = frame.shape[:2]

    # 화면 중앙 크롭
    y1, y2 = int(h*0.25), int(h*0.75)
    x1, x2 = int(w*0.15), int(w*0.85)
    roi = frame[y1:y2, x1:x2]

    num = recognize_unload_by_template(roi)

    if debug:
        print("[DBG UNLOAD]", num, "roi=", roi.shape)

    return num
