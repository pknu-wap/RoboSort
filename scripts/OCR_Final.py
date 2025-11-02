import os
import re
import cv2
import sys
import time
import torch
import argparse
import numpy as np
import json
import threading
import concurrent.futures as futures
from collections import deque, Counter
from typing import Tuple, Optional, List

import serial
import serial.tools.list_ports as list_ports

DETECT_EVERY = 3        # 매 N 프레임마다 탐지
DETECT_WIDTH = 640      # 탐지용 축소 너비
OCR_MIN_INTERVAL = 0.25 # OCR 최소 간격(초)
ROI_DIFF_THRESH = 6.0   # ROI 변동 감지 임계값
TOP_BAND_RATIO = 0.55   # ROI 상단 몇 % 영역 OCR
VOTE_LEN = 7            # 표결 버퍼 길이
DEFAULT_VOTE_MIN = 2    # 표결 최소 횟수
CONF_THRESH = 0.30      # 탐지 score 임계값
BAUD_RATE = 9600        # 시리얼 보드레이트 (아두이노 스케치와 동일)

# NanoDet 경로
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DEFAULT_CONFIG_PATH = os.path.join(project_root, "nanodet_waybill.yml")
DEFAULT_MODEL_PATH  = os.path.join(project_root, "nanodet", "workspace", "waybill", "model_last.ckpt")

CODE_REGEX = re.compile(r"\b\d{3}\s?[A-Z]\d{2}\b")  # 운송장 번호 정규식
ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ " # OCR 허용 문자

# 회전 OCR
ROT_BASE_ANGLES  = [0.0, 180.0]               # 1차 시도
ROT_EXTRA_ANGLES = [90.0, -90.0, 45.0, -45.0] # 2차(품질 낮을 때만)
ROT_SCORE_THR    = 0.65                        # 평균 스코어 임계값(보팅 후보 허용)
IMMEDIATE_SCORE_THR = 0.90                     # 평균 스코어 임계값(즉시 확정 & 전송)

# 경로 지정 (원 코드 유지)
nanodet_path = os.path.join(project_root, 'nanodet')
if nanodet_path not in sys.path:
    sys.path.insert(0, nanodet_path)

from nanodet.util import cfg, load_config, Logger
from demo.demo import Predictor
import rapidocr_onnxruntime

class MatplotlibViewer:
    """OpenCV imshow 대신 Matplotlib로 프레임 표시"""
    def __init__(self, title="RoboSort - Waybill OCR"):
        import matplotlib
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
        self.plt = plt
        self.stop = False
        self.fig, self.ax = plt.subplots()
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        self.im = None
        self.ax.axis("off")
        self.cid = self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        plt.ion(); plt.show()

    def _on_key(self, event):
        if event.key in ("q", "escape"):
            self.stop = True

    def update(self, bgr_img: np.ndarray):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        if self.im is None:
            self.im = self.ax.imshow(rgb)
        else:
            self.im.set_data(rgb)
        self.plt.pause(0.001)

    def close(self):
        try:
            self.fig.canvas.mpl_disconnect(self.cid)
        except Exception: pass
        self.plt.ioff(); self.plt.close(self.fig)

# ----------------------- 전처리/유틸 -----------------------
def preprocess_otsu(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    if max(h, w) < 320:
        bgr = cv2.resize(bgr, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)

def preprocess_adaptive(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    if max(h, w) < 320:
        bgr = cv2.resize(bgr, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 10)

def maybe_invert(bin_img: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(bin_img) if (bin_img > 127).mean() < 0.45 else bin_img

def clamp_box(x1, y1, x2, y2, w, h, margin_ratio=0.05):
    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * margin_ratio); my = int(bh * margin_ratio)
    x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
    x2 = min(w - 1, x2 + mx); y2 = min(h - 1, y2 + my)
    return x1, y1, x2, y2

def roi_changed(prev: Optional[np.ndarray], cur: np.ndarray, thresh: float = ROI_DIFF_THRESH) -> bool:
    if prev is None or prev.size == 0: return True
    a = cv2.resize(prev, (160, 80)); b = cv2.resize(cur, (160, 80))
    if a.ndim == 3: a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim == 3: b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(a, b).mean()) > thresh

def crop_top_band(roi_bgr: np.ndarray, ratio: float = TOP_BAND_RATIO) -> np.ndarray:
    h = roi_bgr.shape[0]; hh = max(10, int(h * ratio))
    return roi_bgr[:hh, :]

def canon(code: str) -> str:
    return re.sub(r"\s+", " ", code.strip().upper())

# ================= NanoDet 결과 파싱 =================
def extract_boxes(nanodet_result) -> List[Tuple[float, int, int, int, int]]:
    boxes = []
    r = nanodet_result
    try:
        if isinstance(r, dict):
            for _, arr in r.items():
                arr = np.array(arr)
                if arr.ndim == 2 and arr.shape[1] >= 5:
                    for row in arr:
                        x1, y1, x2, y2, score = row[:5]
                        boxes.append((float(score), int(x1), int(y1), int(x2), int(y2)))
            return boxes
        if isinstance(r, (list, tuple)):
            for row in r:
                if hasattr(row, "__len__") and len(row) >= 5:
                    x1, y1, x2, y2, score = row[:5]
                    boxes.append((float(score), int(x1), int(y1), int(x2), int(y2)))
            return boxes
        if isinstance(r, np.ndarray) and r.ndim == 2 and r.shape[1] >= 5:
            for row in r:
                x1, y1, x2, y2, score = row[:5]
                boxes.append((float(score), int(x1), int(y1), int(x2), int(y2)))
            return boxes
    except Exception:
        pass
    return boxes

# ================= 밴드/토큰 분리 =================
def pick_text_band(bin_img: np.ndarray) -> np.ndarray:
    black = (bin_img < 128).astype(np.uint8)
    proj = black.mean(axis=1)
    k = max(3, bin_img.shape[0] // 80)
    smooth = np.convolve(proj, np.ones(k, np.float32)/k, mode="same")
    th = max(0.12, float(smooth.mean() * 0.9))
    mask = smooth > th
    if not mask.any(): return bin_img
    ys = np.where(mask)[0]
    splits = np.split(ys, np.where(np.diff(ys) != 1)[0] + 1)
    band = max(splits, key=len)
    y1, y2 = int(band[0]), int(band[-1])
    pad = max(2, (y2 - y1)//6)
    y1 = max(0, y1 - pad); y2 = min(bin_img.shape[0]-1, y2 + pad)
    return bin_img[y1:y2+1, :]

def trim_lr(img: np.ndarray) -> np.ndarray:
    col = (img < 128).mean(axis=0)
    nz = np.where(col > 0.02)[0]
    if nz.size == 0: return img
    return img[:, nz[0]:nz[-1]+1]

def split_letter_digits_cut(token: np.ndarray) -> Optional[int]:
    black = (token < 128).astype(np.uint8)
    proj = black.mean(axis=0); W = token.shape[1]
    if W < 10: return None
    left = max(1, int(W*0.15)); right = min(W-2, int(W*0.85))
    k = max(3, W//30)
    smooth = np.convolve(proj, np.ones(k, np.float32)/k, mode="same")
    cut = int(np.argmin(smooth[left:right]) + left)
    if cut <= 1 or cut >= W-2: return None
    return cut

def split_by_spaces(bin_band: np.ndarray) -> Optional[List[np.ndarray]]:
    black = (bin_band < 128).astype(np.uint8)
    proj = black.mean(axis=0)
    th = max(0.03, float(proj.mean() * 0.5))
    gap_mask = proj < th
    xs = np.where(gap_mask)[0]
    if xs.size == 0: return None
    splits = np.split(xs, np.where(np.diff(xs) != 1)[0] + 1)
    gaps = sorted(splits, key=len, reverse=True)
    if len(gaps) >= 2:
        g = sorted([(int(s[0]), int(s[-1])) for s in gaps[:2]])
        c1 = (g[0][0] + g[0][1]) // 2
        c2 = (g[1][0] + g[1][1]) // 2
        a = bin_band[:, :c1]; b = bin_band[:, c1:c2]; c = bin_band[:, c2:]
        return [trim_lr(a), trim_lr(b), trim_lr(c)]
    g0 = (int(gaps[0][0]), int(gaps[0][-1]))
    cmid = (g0[0] + g0[1]) // 2
    left = trim_lr(bin_band[:, :cmid]); right = trim_lr(bin_band[:, cmid:])
    cut = split_letter_digits_cut(right)
    if cut is None: return None
    return [left, trim_lr(right[:, :cut]), trim_lr(right[:, cut:])]

# ---------------- RapidOCR 모델 경로 -----------------
try:
    rapidocr_dir = os.path.dirname(rapidocr_onnxruntime.__file__)
    det_model_path = os.path.join(rapidocr_dir, "models", "ch_PP-OCRv4_det_infer.onnx")
    rec_model_path = os.path.join(rapidocr_dir, "models", "ch_PP-OCRv4_rec_infer.onnx")
    cls_model_path = os.path.join(rapidocr_dir, "models", "ch_ppocr_mobile_v2.0_cls_infer.onnx")
    if not all(os.path.exists(p) for p in [det_model_path, rec_model_path, cls_model_path]):
        print("[WARN] Default ONNX models not found in rapidocr_onnxruntime package.")
        det_model_path, rec_model_path, cls_model_path = None, None, None
except (ImportError, AttributeError):
    print("[WARN] rapidocr_onnxruntime not found or `__file__` not available.")
    det_model_path, rec_model_path, cls_model_path = None, None, None

class RapidRec:
    def __init__(self):
        if det_model_path and rec_model_path and cls_model_path:
            self.ocr = rapidocr_onnxruntime.RapidOCR(
                det_model_path=det_model_path,
                rec_model_path=rec_model_path,
                cls_model_path=cls_model_path,
                rec_score_threshold=0.1
            )
        else:
            self.ocr = rapidocr_onnxruntime.RapidOCR(rec_score_threshold=0.1)

    def _run(self, img: np.ndarray, allow: str) -> Tuple[str, float]:
        res, _ = self.ocr(img, use_det=False, use_cls=True)
        txt, score = "", 0.0
        try:
            if isinstance(res, list) and len(res) > 0:
                item = res[0]
                if isinstance(item, (list, tuple)):
                    if len(item) == 2 and isinstance(item[0], str):
                        txt, score = item[0], float(item[1])
                    elif len(item) >= 3 and isinstance(item[1], str):
                        txt, score = item[1], float(item[2])
        except Exception:
            pass
        txt = "".join(ch for ch in txt.upper() if ch in ALLOWLIST)
        return txt, float(score)

    def digits(self, img: np.ndarray) -> str:
        t, _ = self._run(img, "0123456789"); return t

    def letters(self, img: np.ndarray) -> str:
        t, _ = self._run(img, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"); return t

    def any_en(self, img: np.ndarray) -> str:
        t, _ = self._run(img, ALLOWLIST); return t

    def detect_full(self, img: np.ndarray) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        try:
            res, _ = self.ocr(img, use_det=True, use_cls=True)
            for item in (res or []):
                if isinstance(item, (list, tuple)) and len(item) >= 3 and isinstance(item[1], str):
                    txt = "".join(ch for ch in item[1].upper() if ch in ALLOWLIST)
                    if txt:
                        out.append((txt, float(item[2])))
        except Exception:
            pass
        return out

rapid = RapidRec()  # 전역 하나로 재사용(워밍업됨)

# ---------------- OCR 세부 ----------------
def split_three_chars(bin_img: np.ndarray) -> Optional[List[np.ndarray]]:
    black = (bin_img < 128).astype(np.uint8)
    proj = black.mean(axis=0); W = bin_img.shape[1]
    if W < 24: return None
    k = max(3, W // 30)
    smooth = np.convolve(proj, np.ones(k, np.float32)/k, mode="same")
    cuts = []
    for i in (1, 2):
        ideal = int(W * i / 3)
        l = max(1, ideal - max(3, W // 10))
        r = min(W-2, ideal + max(3, W // 10))
        cuts.append(int(np.argmin(smooth[l:r]) + l))
    a = trim_lr(bin_img[:, :cuts[0]]); b = trim_lr(bin_img[:, cuts[0]:cuts[1]]); c = trim_lr(bin_img[:, cuts[1]:])
    return [a, b, c]

def read_left_digits_strict(left_bin: np.ndarray) -> Optional[str]:
    if left_bin is None or left_bin.size == 0:
        return None
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    left_bin = cv2.morphologyEx(left_bin, cv2.MORPH_CLOSE, k, iterations=1)
    chars = split_three_chars(left_bin)
    if not chars or len(chars) != 3: return None
    out = []
    for ch in chars:
        d = rapid.digits(ch)
        if not d: return None
        out.append(d[0])
    return "".join(out)

def ocr_code_segmented(roi_bgr: np.ndarray, debug=False) -> Tuple[Optional[str], bool]:
    if roi_bgr is None or roi_bgr.size == 0:
        return None, False
    head = crop_top_band(roi_bgr, TOP_BAND_RATIO)
    bins = []
    for pre in (preprocess_adaptive, preprocess_otsu):
        b = pre(head); b = maybe_invert(b); bins.append(b)

    for bin_img in bins:
        band = pick_text_band(bin_img)
        parts = split_by_spaces(band)
        if not parts or len(parts) != 3:
            continue
        left, mid, right = parts
        if left.size == 0 or mid.size == 0 or right.size == 0:
            continue

        left_digits = read_left_digits_strict(left)
        strict_ok = left_digits is not None and len(left_digits) >= 3
        if not strict_ok:
            left_digits = rapid.digits(left)[:3]
        if not left_digits or len(left_digits) < 3:
            continue

        mid_char = rapid.letters(mid)[:1]
        if len(mid_char) < 1:
            continue

        right_digits = rapid.digits(right)[:2]
        if len(right_digits) < 2:
            continue

        cand = f"{left_digits[:3]} {mid_char[0]}{right_digits[:2]}"
        if CODE_REGEX.fullmatch(cand):
            if debug: print(f"[DEBUG] segmented(rapid) -> {cand} (strict={strict_ok})")
            return cand, True  # 분할 성공은 즉시 확정

    # 전체 한 번 읽고 정규화(폴백)
    merged = rapid.any_en(head)
    s = canon(merged)
    m = re.search(r"(\d{3})\s*([A-Z])\s*(\d{2})", s)
    if m:
        cand = f"{m.group(1)} {m.group(2)}{m.group(3)}"
        if CODE_REGEX.fullmatch(cand):
            if debug: print(f"[DEBUG] fallback(rapid) -> {cand}")
            return cand, False
    return None, False

# -------------------- 회전 유틸/로직 --------------------
def rotate_bound(img: np.ndarray, angle_deg: float) -> np.ndarray:
    if angle_deg % 360 == 0:
        return img
    (h, w) = img.shape[:2]
    cX, cY = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def ocr_code_rotation_adaptive(
    roi_bgr: np.ndarray,
    debug: bool = False,
    base_angles: Optional[List[float]] = None,
    extra_angles: Optional[List[float]] = None,
    score_thr: float = ROT_SCORE_THR,
    immediate_thr: float = IMMEDIATE_SCORE_THR,
) -> Tuple[Optional[str], bool]:
    """
    1) base_angles(기본 0,180)에서 분할식 우선 → 성공 시 즉시(True)
    2) base_angles에서 검출 기반 평균 스코어 계산:
       - mean >= immediate_thr → 즉시 확정(True)
       - mean >= score_thr     → 보팅 후보(False)
    3) 품질 낮으면 extra 각도(90,-90,±45)에서 동일 로직
    """
    base_angles = base_angles or ROT_BASE_ANGLES
    extra_angles = extra_angles or ROT_EXTRA_ANGLES

    # 1) base에서 분할식 → 성공 시 즉시
    for ang in base_angles:
        img = roi_bgr if ang == 0.0 else rotate_bound(roi_bgr, ang)
        code, immediate = ocr_code_segmented(img, debug)
        if code:
            return code, True  # 분할 성공은 즉시

    # 1-b) base에서 검출 기반
    best_cand, best_score = None, -1.0
    for ang in base_angles:
        img = roi_bgr if ang == 0.0 else rotate_bound(roi_bgr, ang)
        items = rapid.detect_full(img)
        if not items:
            continue
        combined = " ".join([t for (t, s) in items])
        s = canon(combined)
        m = re.search(r"(\d{3})\s*([A-Z])\s*(\d{2})", s)
        if m:
            cand = f"{m.group(1)} {m.group(2)}{m.group(3)}"
            mean_score = float(np.mean([s for (_, s) in items]))
            if debug:
                print(f"[DEBUG] base{ang:+.0f} -> {cand} (mean={mean_score:.3f})")
            if mean_score >= immediate_thr:
                return cand, True
            if mean_score > best_score:
                best_score, best_cand = mean_score, cand

    if best_cand and best_score >= score_thr:
        return best_cand, False

    # 2) extra 각도
    best_cand2, best_score2 = None, -1.0
    for ang in extra_angles:
        img = rotate_bound(roi_bgr, ang)
        items = rapid.detect_full(img)
        if not items:
            continue
        combined = " ".join([t for (t, s) in items])
        s = canon(combined)
        m = re.search(r"(\d{3})\s*([A-Z])\s*(\d{2})", s)
        if m:
            cand = f"{m.group(1)} {m.group(2)}{m.group(3)}"
            mean_score = float(np.mean([s for (_, s) in items]))
            if debug:
                print(f"[DEBUG] extra{ang:+.0f} -> {cand} (mean={mean_score:.3f})")
            if mean_score >= immediate_thr:
                return cand, True
            if mean_score > best_score2:
                best_score2, best_cand2 = mean_score, cand

    if best_cand2:
        return best_cand2, False
    return None, False

def resize_for_detect(frame: np.ndarray, det_w: int):
    h, w = frame.shape[:2]
    if w <= det_w: return frame, 1.0
    scale = det_w / float(w)
    small = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return small, scale

# ===================== 구역/시리얼 유틸 =====================
def code_to_zone(code: str) -> Optional[int]:
    """ '619 A02' -> 600  (앞 3자리의 백의 자리로 내림 ×100) """
    m = re.search(r"(\d{3})\s?[A-Z]\d{2}", code.upper())
    if not m: return None
    abc = int(m.group(1))
    return (abc // 100) * 100

def find_serial_port(prefer: Optional[str]=None) -> Optional[str]:
    if prefer:
        return prefer
    ports = list_ports.comports()
    for p in ports:
        name = (p.device or "") + " " + (p.description or "")
        if any(k in name for k in ("Arduino", "wchusb", "ttyACM", "ttyUSB", "COM")):
            return p.device
    return ports[0].device if ports else None

class SerialSender:
    """ 아두이노로 '세 자리 코드(예: 900, 500, 600...)' 전송 """
    def __init__(self, port: Optional[str], baud: int, dup_policy: str = "handshake", disable_serial: bool = False):
        self.port_name = find_serial_port(port)
        self.baud = baud
        self.ser = None
        self.last_sent = None 
        self.dup_policy = dup_policy
        self._stop_reader = False
        self._reader_th: Optional[threading.Thread] = None

        if disable_serial:
            print("[INFO] Serial communication is disabled by --disable_serial.")
            return

        if self.port_name:
            try:
                self.ser = serial.Serial(self.port_name, self.baud, timeout=0.1)
                time.sleep(1.5)  # Uno 리셋 대기
                print(f"[INFO] Serial opened: {self.port_name} @ {self.baud}")
                if self.dup_policy == "handshake":
                    self._reader_th = threading.Thread(target=self._reader_loop, daemon=True)
                    self._reader_th.start()
            except Exception as e:
                print(f"[WARN] Serial open failed ({self.port_name}): {e}")
                self.ser = None
        else:
            print("[WARN] No serial port found. Running without sending.")

    def _reader_loop(self):
        while not self._stop_reader and self.ser is not None:
            try:
                line = self.ser.readline()
                if not line:
                    continue
                text = line.decode("utf-8", errors="ignore").strip()
                if text:
                    print(f"[SER] {text}")
                    if "READY" in text or "[STATE] IDLE" in text:
                        self.last_sent = None
            except Exception:
                time.sleep(0.05)

    def send_code(self, code: int):
        """ 세 자리 정수 코드를 전송 """
        if code is None:
            return
        if self.ser is None:
            print(f"[SEND-DRY] {code}")
            return

        if self.dup_policy == "block" and self.last_sent == code:
            return
        if self.dup_policy == "handshake" and self.last_sent == code:
            return

        line = f"{int(code)}\n".encode()
        try:
            self.ser.write(line)
            self.ser.flush()
            print(f"[SEND] {line!r} -> {self.port_name}")
            self.last_sent = int(code)
        except Exception as e:
            print(f"[WARN] Serial send failed: {e}")
            self.ser = None

    def close(self):
        try:
            self._stop_reader = True
            if self._reader_th and self._reader_th.is_alive():
                self._reader_th.join(timeout=0.2)
            if self.ser: self.ser.close()
        except Exception:
            pass

# ---------------------- OCR 로직 ----------------------
def ocr_with_rotation(roi_bgr: np.ndarray, debug: bool, rot_disabled: bool) -> Tuple[Optional[str], bool]:
    if rot_disabled:
        return ocr_code_segmented(roi_bgr, debug)
    else:
        return ocr_code_rotation_adaptive(
            roi_bgr, debug,
            base_angles=ROT_BASE_ANGLES,
            extra_angles=ROT_EXTRA_ANGLES,
            score_thr=ROT_SCORE_THR,
            immediate_thr=IMMEDIATE_SCORE_THR
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camid", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--debug_ocr", action="store_true")

    # 시리얼 관련
    parser.add_argument("--serial_port", type=str, default=None, help="예: /dev/ttyACM0 또는 COM3 (미지정 시 자동)")
    parser.add_argument("--dup_policy", type=str, default="handshake",
                        choices=["handshake","always","block"],
                        help="중복 명령 처리 정책 (기본 handshake: READY/[STATE] IDLE 수신 시 중복 허용)")
    parser.add_argument("--disable_serial", action="store_true",
                        help="시리얼 통신 비활성화(로그만 출력)")

    # 영상 처리
    parser.add_argument("--hflip_input", action="store_true",
                        help="입력 프레임을 좌우반전(처리와 표시 모두 반전된 영상 기준)")

    # 회전 OCR 토글
    parser.add_argument("--rot_disable", action="store_true",
                        help="회전 견고 OCR 비활성화(기존 segmented/fallback만 사용)")

    args = parser.parse_args()
    VOTE_MIN = DEFAULT_VOTE_MIN

    # NanoDet
    load_config(cfg, DEFAULT_CONFIG_PATH)
    logger = Logger(-1, use_tensorboard=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = Predictor(cfg, DEFAULT_MODEL_PATH, logger, device=device)

    # Camera
    cap = cv2.VideoCapture(args.camid, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    viewer = MatplotlibViewer("RoboSort - Waybill OCR") if args.show else None

    executor = futures.ThreadPoolExecutor(max_workers=1)
    pending: Optional[futures.Future] = None
    last_ocr_t = 0.0
    vote_buf = deque(maxlen=VOTE_LEN)
    last_emitted = ""
    prev_roi: Optional[np.ndarray] = None
    streak_code = ""
    streak_cnt = 0

    frame_idx = 0
    best_box = None

    sender = SerialSender(args.serial_port, BAUD_RATE, dup_policy=args.dup_policy, disable_serial=args.disable_serial)

    print("[INFO] Press 'q' or close window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] camera read failed."); break

            if args.hflip_input:
                frame = cv2.flip(frame, 1)

            # 1) 탐지
            need_detect = (frame_idx % max(1, DETECT_EVERY) == 0) or (best_box is None)
            vis = frame
            if need_detect:
                det_img, scale = resize_for_detect(frame, DETECT_WIDTH)
                meta, res = predictor.inference(det_img)
                boxes = extract_boxes(res[0])
                chosen = None
                for score, x1, y1, x2, y2 in boxes:
                    if score < CONF_THRESH: continue
                    x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
                    area = max(0,(x2-x1))*max(0,(y2-y1))
                    key = (score, area)
                    if chosen is None or key > chosen[0]:
                        chosen = (key, (x1, y1, x2, y2, score))
                best_box = chosen[1] if chosen is not None else None

            # 2) ROI
            roi = None
            if best_box is not None:
                h, w = frame.shape[:2]
                x1, y1, x2, y2, score = best_box
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h, 0.05)
                roi = frame[y1:y2, x1:x2]
                if viewer is not None:
                    vis = frame.copy()
                    cv2.rectangle(vis, (x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(vis, f"waybill {score:.2f}", (x1, max(0,y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

            # 3) OCR 제출(ROI 변동 없어도 표결 미완이면 계속)
            now = time.time()
            need_more_votes = len(vote_buf) < VOTE_MIN
            if roi is not None and (now - last_ocr_t) >= OCR_MIN_INTERVAL:
                if roi_changed(prev_roi, roi) or need_more_votes:
                    if pending is None or pending.done():
                        def worker(img, dbg, rot_off):
                            return ocr_with_rotation(img, dbg, rot_off)
                        pending = executor.submit(worker, roi.copy(), args.debug_ocr, args.rot_disable)
                        last_ocr_t = now
                        prev_roi = roi.copy()

            # 4) 결과 수거 + 즉시/연속/다수결 + 시리얼 전송
            if pending is not None and pending.done():
                (code, immediate) = pending.result()
                pending = None
                if code:
                    code = canon(code)

                    if immediate:
                        print(code)
                        last_emitted = code
                        vote_buf.clear()
                        streak_code, streak_cnt = "", 0
                    else:
                        if code == streak_code:
                            streak_cnt += 1
                        else:
                            streak_code, streak_cnt = code, 1
                        if args.debug_ocr:
                            print(f"[VOTE] streak {streak_cnt}/{VOTE_MIN}: {streak_code}")
                        if streak_cnt >= VOTE_MIN and code != last_emitted:
                            print(code)
                            last_emitted = code
                            vote_buf.clear()
                            streak_code, streak_cnt = "", 0
                        else:
                            vote_buf.append(code)
                            common, cnt = Counter(vote_buf).most_common(1)[0]
                            if args.debug_ocr:
                                print(f"[VOTE] top={common} ({cnt}), buf={len(vote_buf)}")
                            if cnt >= VOTE_MIN and common != last_emitted:
                                print(common)
                                last_emitted = common
                                streak_code, streak_cnt = "", 0

                    if last_emitted:
                        zone = code_to_zone(last_emitted)  # 619A02 -> 600
                        if zone is not None:
                            sender.send_code(zone)
                        else:
                            print("[WARN] code->zone failed:", last_emitted)

            # 5) 뷰
            if viewer is not None:
                viewer.update(vis)
                if viewer.stop: break

            frame_idx += 1

    finally:
        cap.release()
        if viewer is not None: viewer.close()
        try: cv2.destroyAllWindows()
        except: pass
        sender.close()

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    sys.exit(main())
