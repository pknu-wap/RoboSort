import argparse, json, os, shutil, random, glob
from pathlib import Path

# Converts YOLO labels to COCO and splits into train/val.
# Assumes one class 'waybill' with id 0.

def yolo_to_coco(images_dir, labels_dir, out_json, copy_images_dir=None):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    images = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])

    coco = {
        "info": {"description": "Waybill dataset (YOLO->COCO)"},
        "licenses": [],
        "categories": [{"id":1, "name":"waybill", "supercategory":"object"}],
        "images": [],
        "annotations": []
    }
    ann_id = 1
    for img_id, img_path in enumerate(images, start=1):
        # try to read image size via cv2
        import cv2
        im = cv2.imread(str(img_path))
        if im is None:
            print(f"WARNING: cannot read {img_path}")
            continue
        h, w = im.shape[:2]

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "height": h,
            "width": w
        })

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            # no labels: skip annotations
            continue
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, bw, bh = map(float, parts)
                # YOLO normalized -> COCO bbox [x,y,w,h] in pixels
                x = (xc - bw/2) * w
                y = (yc - bh/2) * h
                ww = bw * w
                hh = bh * h
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x, y, ww, hh],
                    "area": ww*hh,
                    "iscrowd": 0
                })
                ann_id += 1

        # Optionally copy images to a flat folder (for NanoDet config convenience)
        if copy_images_dir:
            os.makedirs(copy_images_dir, exist_ok=True)
            shutil.copy2(img_path, os.path.join(copy_images_dir, img_path.name))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="dataset", help="folder containing images/ and labels/")
    ap.add_argument("--split", type=float, default=0.9, help="train ratio")
    ap.add_argument("--copy", action="store_true", help="copy images to dataset/coco/images")
    args = ap.parse_args()

    root = Path(args.data_root)
    images = sorted([p for p in (root/"images").glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    random.seed(0)
    random.shuffle(images)
    n_train = int(len(images)*args.split)
    train_imgs = set(p.name for p in images[:n_train])

    coco_dir = root/"coco"
    img_out = coco_dir/"images" if args.copy else None
    os.makedirs(coco_dir, exist_ok=True)
    if img_out: os.makedirs(img_out, exist_ok=True)

    # Write train/val json
    def filter_and_make(split_name, names_set):
        # Temporarily create filtered views by copying only names? We'll generate jsons that include only those images.
        # We'll just restrict in the converter by checking membership.
        pass

    # We'll run conversion twice by toggling membership checks
    def convert_subset(subset_names, out_json):
        # Create temporary views by moving non-subset labels? Instead, we'll just copy to temp folders.
        tmp_images = coco_dir/f"tmp_images_{out_json.stem}"
        tmp_labels = coco_dir/f"tmp_labels_{out_json.stem}"
        if tmp_images.exists():
            shutil.rmtree(tmp_images)
        if tmp_labels.exists():
            shutil.rmtree(tmp_labels)
        os.makedirs(tmp_images, exist_ok=True)
        os.makedirs(tmp_labels, exist_ok=True)
        for p in (root/"images").glob("*"):
            if p.name in subset_names:
                shutil.copy2(p, tmp_images/p.name)
                label = (root/"labels"/(p.stem+".txt"))
                if label.exists():
                    shutil.copy2(label, tmp_labels/label.name)
        yolo_to_coco(tmp_images, tmp_labels, out_json, copy_images_dir=(img_out if args.copy else None))
        shutil.rmtree(tmp_images)
        shutil.rmtree(tmp_labels)

    convert_subset(train_imgs, coco_dir/"train.json")
    val_names = set(p.name for p in images) - train_imgs
    convert_subset(val_names, coco_dir/"val.json")
    print("COCO annotations written to:", coco_dir)

if __name__ == "__main__":
    main()