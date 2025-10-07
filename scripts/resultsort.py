# 다운 받은 이미지 및 라벨 txt 파일을 train/val 폴더로 나누기

import os, shutil, random

SRC_IMAGE_DIR = "dataset/images"
SRC_LABEL_DIR = "dataset/labels"
OUT_DIR = "dataset"
SPLIT_RATIO = 0.9  # 90% train

os.makedirs(f"{OUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/images/val", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/val", exist_ok=True)

images = [f for f in os.listdir(SRC_IMAGE_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))]
random.shuffle(images)

split = int(len(images) * SPLIT_RATIO)
train_images = images[:split]
val_images = images[split:]

def move_files(file_list, split_type):
    for img in file_list:
        base = os.path.splitext(img)[0]
        label_txt = f"{base}.txt"
        shutil.copy(f"{SRC_IMAGE_DIR}/{img}", f"{OUT_DIR}/images/{split_type}/{img}")
        if os.path.exists(f"{SRC_LABEL_DIR}/{label_txt}"):
            shutil.copy(f"{SRC_LABEL_DIR}/{label_txt}", f"{OUT_DIR}/labels/{split_type}/{label_txt}")

move_files(train_images, "train")
move_files(val_images, "val")

print(f"✅ Done. Train: {len(train_images)}, Val: {len(val_images)}")
