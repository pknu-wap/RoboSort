import cv2
import os
import numpy as np
import random

INPUT_DIR = "C:\\Users\\wave\\OneDrive\\Desktop\\data"
OUTPUT_DIR = "C:\\Users\\wave\\OneDrive\\Desktop\\data_augmented"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# 1) Motion Blur
# ==============================
def motion_blur(img, degree=10, angle=0):
    degree = max(1, degree)
    kernel = np.zeros((degree, degree))
    kernel[int((degree - 1) / 2), :] = np.ones(degree)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1.0), (degree, degree))
    kernel = kernel / degree
    return cv2.filter2D(img, -1, kernel)

# ==============================
# 2) Perspective (우그러짐)
# ==============================
def perspective_warp(img, max_shift=0.08):
    h, w = img.shape[:2]

    shift_w = w * max_shift
    shift_h = h * max_shift

    pts1 = np.float32([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h]
    ])

    pts2 = np.float32([
        [random.uniform(0, shift_w), random.uniform(0, shift_h)],
        [w - random.uniform(0, shift_w), random.uniform(0, shift_h)],
        [random.uniform(0, shift_w), h - random.uniform(0, shift_h)],
        [w - random.uniform(0, shift_w), h - random.uniform(0, shift_h)]
    ])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped

# ==============================
# 3) Gamma Correction (조명 보정)
# ==============================
def gamma_correction(img, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

# ==============================
# 4) Soft Gaussian Blur (Mild)
# ==============================
def soft_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def rotate_360(img):
    angle = random.uniform(0, 360)

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # 회전 후 캔버스를 확장하여 잘림 방지
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    # 새로운 bounding box 크기 계산
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 중심 맞추기 보정
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(img, rotation_matrix, (new_w, new_h))
    return rotated, angle


# ==============================
# Main Loop
# ==============================
for filename in os.listdir(INPUT_DIR):
    if not (filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg")):
        continue

    img_path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(img_path)

    if img is None:
        print("이미지 읽기 오류:", filename)
        continue

    name = os.path.splitext(filename)[0]

    # 기본 저장
    cv2.imwrite(f"{OUTPUT_DIR}/{name}_orig.jpg", img)

    # ----------------------------------------
    # 1. Motion Blur (OCR 흔들린 카메라 대비)
    # ----------------------------------------
    mb = motion_blur(img, degree=random.randint(8, 15), angle=random.randint(0, 180))
    cv2.imwrite(f"{OUTPUT_DIR}/{name}_motion.jpg", mb)

    # ----------------------------------------
    # 2. Perspective (운송장 기울어지거나 우그러짐)
    # ----------------------------------------
    persp = perspective_warp(img, max_shift=0.06)
    cv2.imwrite(f"{OUTPUT_DIR}/{name}_persp.jpg", persp)

    # ----------------------------------------
    # 3. Gamma Correction (어두운 조명 / 밝은 조명 대비)
    # ----------------------------------------
    gamma_dark = gamma_correction(img, 0.6)
    gamma_bright = gamma_correction(img, 1.6)
    cv2.imwrite(f"{OUTPUT_DIR}/{name}_gamma_dark.jpg", gamma_dark)
    cv2.imwrite(f"{OUTPUT_DIR}/{name}_gamma_bright.jpg", gamma_bright)

    # ----------------------------------------
    # 4. Soft Gaussian Blur (카메라 포커스 미세 오차 용)
    # ----------------------------------------
    blur = soft_blur(img)
    cv2.imwrite(f"{OUTPUT_DIR}/{name}_softblur.jpg", blur)

    for i in range(10):
        rotated, angle = rotate_360(img)
        cv2.imwrite(f"{OUTPUT_DIR}/{name}_rot_{i}_{int(angle)}.jpg", rotated)
    print("완료:", filename)

print("=== 모든 OCR 최적화 증강 완료! ===")
