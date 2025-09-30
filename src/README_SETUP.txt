NANODET WAYBILL PROJECT — CPU training + ONNXRuntime(DML) inference
===================================================================

This package gives you a working pipeline:
1) Convert your YOLO-format labels to COCO for NanoDet
2) Train NanoDet (CPU easiest; optional DirectML for training if you patch upstream)
3) Export to ONNX
4) Run fast inference with ONNX Runtime (DirectML provider on Intel Arc)
5) Crop detected waybill block, run Tesseract OCR with regex, and send the code via Serial or UDP

Directory you should create (example):
--------------------------------------
project_root/
 ├─ dataset/
 │   ├─ images/              # put ALL images here (jpg/png)
 │   ├─ labels/              # YOLO *.txt labels here (same basename as images)
 ├─ nanodet_waybill_config_example.yml
 ├─ prepare_dataset.py
 ├─ detect_onnx_tesseract.py
 └─ README_SETUP.txt

STEP 0) Create venv (recommended)
---------------------------------
# PowerShell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

STEP 1) Install dependencies
----------------------------
# Minimal training (CPU) + tools
pip install -U pip wheel
pip install opencv-python numpy pillow tqdm matplotlib pyyaml scikit-learn
pip install pycocotools-windows  # if Windows; otherwise: pip install pycocotools
pip install pytesseract
# OnnxRuntime with DirectML for fast inference on Intel Arc
pip install onnxruntime-directml

# OPTIONAL: if you want to try training with DirectML (experimental):
pip install torch-directml
# Otherwise for CPU-only training:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

STEP 2) Clone NanoDet
---------------------
git clone https://github.com/nanodet/nanodet.git
cd nanodet
pip install -r requirements.txt
cd ..

(If any build fails for COCO API, you already installed pycocotools-windows above.)

STEP 3) Prepare dataset (split + convert YOLO -> COCO)
------------------------------------------------------
# Assuming your raw images/labels are under dataset/images, dataset/labels
python prepare_dataset.py --data-root dataset --split 0.9

This creates:
dataset/
 ├─ images/
 ├─ labels/
 ├─ coco/
 │   ├─ train.json
 │   ├─ val.json
 │   └─ images/ symlinks or copies (see script flag)

STEP 4) Copy+edit config
------------------------
Use nanodet_waybill_config_example.yml as a starting point.
Open it and ensure the paths under 'data' point to your dataset/coco/*.json and image dirs.

STEP 5) Train (CPU shown here)
------------------------------
cd nanodet
python tools/train.py ../nanodet_waybill_config_example.yml --device cpu
# (If you patched NanoDet to use torch-directml, you could use: --device dml)

Outputs go to: nanodet/workspace/ (by default: ./workspace/waybill)

STEP 6) Export to ONNX
----------------------
# Replace BEST_EPOCH.pth with your best checkpoint path
python tools/export_onnx.py ../nanodet_waybill_config_example.yml BEST_EPOCH.pth --output ../waybill.onnx

STEP 7) Run real-time inference + OCR + send to Pi/Arduino
-----------------------------------------------------------
# Tesseract must be installed system-wide (Windows installer). Then run:
python detect_onnx_tesseract.py --model waybill.onnx --camera 0 \
   --send udp --udp-host 192.168.0.50 --udp-port 5005 \
   --pattern "(\d{3}\s?[A-Z]\d{2})"

# For Arduino over Serial (change COM port and baud):
python detect_onnx_tesseract.py --model waybill.onnx --camera 0 \
   --send serial --serial-port COM4 --baud 115200

Notes
-----
- We detect only one class: 'waybill' (the number block like '013 A03').
- OCR runs only on the top-scoring detection's crop by default. Use --all to OCR all boxes.
- Use --min-score to filter weak boxes.
- The script prints and overlays recognized code. It also sends it once per stable reading (debounced).