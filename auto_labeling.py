import os
import cv2
import random
import numpy as np
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================
ORIGIN_IMG_DIR = "dataset/images/train"
ORIGIN_LAB_DIR = "dataset/labels/train"

SAVE_IMG_DIR = "dataset_augmented/images/train"
SAVE_LAB_DIR = "dataset_augmented/labels/train"

AUG_PER_IMAGE = 12     # 97장 → 97*12 ≈ 1164장
IMAGE_SIZE = 640       # 최종 학습 이미지 크기

# ==========================
# 유틸 함수
# ==========================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir(SAVE_IMG_DIR)
ensure_dir(SAVE_LAB_DIR)

def load_label(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f.readlines():
            c, x, y, w, h = map(float, line.strip().split())
            boxes.append([c, x, y, w, h])
    return boxes

def save_label(save_path, boxes):
    with open(save_path, "w") as f:
        for b in boxes:
            f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

# ==========================
# 변환 함수들
# ==========================
def random_brightness(img):
    factor = random.uniform(0.6, 1.4)
    img = np.clip(img * factor, 0, 255).astype(np.uint8)
    return img

def random_contrast(img):
    factor = random.uniform(0.7, 1.3)
    mean = np.mean(img)
    img = np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)
    return img

def random_noise(img):
    noise = np.random.normal(0, 8, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    return img

def random_blur(img):
    k = random.choice([3,5,7])
    return cv2.GaussianBlur(img, (k, k), 0)

def rotate_bbox(cx, cy, w, h, angle_rad):
    # YOLO 좌표 기준
    # (cx, cy), (w, h), angle → rotate 후 다시 YOLO 형식으로 변환
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    x1 = cx - w/2; y1 = cy - h/2
    x2 = cx + w/2; y2 = cy + h/2

    points = np.array([
        [x1,y1],[x2,y1],[x1,y2],[x2,y2]
    ])

    R = np.array([[c,-s],[s,c]])
    rotated = points @ R.T

    new_x1 = rotated[:,0].min()
    new_y1 = rotated[:,1].min()
    new_x2 = rotated[:,0].max()
    new_y2 = rotated[:,1].max()

    cx2 = (new_x1 + new_x2)/2
    cy2 = (new_y1 + new_y2)/2
    w2 = new_x2 - new_x1
    h2 = new_y2 - new_y1

    return cx2, cy2, w2, h2

def random_rotate(img, boxes):
    angle = random.uniform(-20, 20)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=(128,128,128))

    angle_rad = -np.radians(angle)  # YOLO 좌표는 반대방향 필요
    new_boxes = []
    for c, x, y, ww, hh in boxes:
        cx = (x - 0.5) * w
        cy = (y - 0.5) * h
        w0 = ww * w
        h0 = hh * h
        cx2, cy2, w2, h2 = rotate_bbox(cx, cy, w0, h0, angle_rad)
        new_boxes.append([
            c,
            cx2/w + 0.5,
            cy2/h + 0.5,
            w2/w,
            h2/h
        ])
    return rotated, new_boxes

# ==========================
# 메인 루프
# ==========================
images = os.listdir(ORIGIN_IMG_DIR)

print(f"[INFO] 총 {len(images)}장의 이미지 증강 시작")
for name in tqdm(images):

    if not name.lower().endswith((".jpg",".png",".jpeg")):
        continue

    img_path = os.path.join(ORIGIN_IMG_DIR, name)
    lab_path = os.path.join(ORIGIN_LAB_DIR, name.replace(".jpg",".txt").replace(".png",".txt"))

    img = cv2.imread(img_path)
    boxes = load_label(lab_path)

    for i in range(AUG_PER_IMAGE):

        aug = img.copy()
        bbs = [b[:] for b in boxes]

        # 랜덤 effect 조합
        if random.random() < 0.8: aug = random_brightness(aug)
        if random.random() < 0.8: aug = random_contrast(aug)
        if random.random() < 0.5: aug = random_noise(aug)
        if random.random() < 0.5: aug = random_blur(aug)
        if random.random() < 0.7: aug, bbs = random_rotate(aug, bbs)

        # 저장
        new_name = name.replace(".jpg", f"_aug{i}.jpg").replace(".png", f"_aug{i}.jpg")
        cv2.imwrite(os.path.join(SAVE_IMG_DIR, new_name), aug)
        save_label(os.path.join(SAVE_LAB_DIR, new_name.replace(".jpg",".txt")), bbs)

print("[INFO] 증강 완료!")
