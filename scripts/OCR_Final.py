# realtime_waybill_ocr_tesseract_fast.py
import os

import re
import cv2
import sys
import time
import torch
import argparse
import numpy as np
import uuid
import shutil
import concurrent.futures as futures
from collections import deque, Counter
from typing import Tuple, Optional, List
from nanodet.util import cfg, load_config, Logger
from demo.demo import Predictor
from rapidocr_onnxruntime import RapidOCR

# 경로 지정
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
nanodet_path = os.path.join(project_root, 'nanodet')
if nanodet_path not in sys.path:
    sys.path.insert(0, nanodet_path)




DETECT_EVERY = 3        # 매 N 프레임마다 탐지
DETECT_WIDTH = 640     # 탐지용 축소 너비
OCR_MIN_INTERVAL = 0.25 # OCR 최소 간격(초)
ROI_DIFF_THRESH = 6.0 # ROI 변동 감지 임계값
TOP_BAND_RATIO = 0.55 # ROI 상단 몇 % 영역 OCR
VOTE_LEN = 7         # 표결 버퍼 길이
DEFAULT_VOTE_MIN = 2 # 표결 최소 횟수

CODE_REGEX = re.compile(r"\b\d{3}\s?[A-Z]\d{2}\b") # 운송장 번호 정규식
ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ " # OCR 허용 문자

# matplotlib 뷰어
class MatplotlibViewer:
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

def preprocess_otsu(bgr: np.ndarray) -> np.ndarray:
    # Otsu 이진화를 사용한 이미지 전처리
    h, w = bgr.shape[:2]
    if max(h, w) < 320:
        bgr = cv2.resize(bgr, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)

def preprocess_adaptive(bgr: np.ndarray) -> np.ndarray:
    # 적응형 스레시홀딩을 사용한 이미지 전처리
    h, w = bgr.shape[:2]
    if max(h, w) < 320:
        bgr = cv2.resize(bgr, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 10)

def maybe_invert(bin_img: np.ndarray) -> np.ndarray:
    # 글자보다 배경이 밝은 경우 이미지 반전
    return cv2.bitwise_not(bin_img) if (bin_img > 127).mean() < 0.45 else bin_img

def clamp_box(x1, y1, x2, y2, w, h, margin_ratio=0.05):
    # 바운딩 박스가 이미지 경계를 벗어나지 않도록 좌표 조정
    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * margin_ratio); my = int(bh * margin_ratio)
    x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
    x2 = min(w - 1, x2 + mx); y2 = min(h - 1, y2 + my)
    return x1, y1, x2, y2

def roi_changed(prev: Optional[np.ndarray], cur: np.ndarray, thresh: float = ROI_DIFF_THRESH) -> bool:
    # 이전 ROI와 현재 ROI를 비교하여 변화 여부 확인
    if prev is None or prev.size == 0: return True
    a = cv2.resize(prev, (160, 80)); b = cv2.resize(cur, (160, 80))
    if a.ndim == 3: a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim == 3: b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(a, b).mean()) > thresh

def crop_top_band(roi_bgr: np.ndarray, ratio: float = TOP_BAND_RATIO) -> np.ndarray:
    # ROI이미지의 상단 일부 영역(운송장 번호가 있을 가능성이 높은) 재단.
    h = roi_bgr.shape[0]; hh = max(10, int(h * ratio))
    return roi_bgr[:hh, :]

def canon(code: str) -> str:
    # 인식된 운송장 번호 문자열의 정규화.
    return re.sub(r"\s+", " ", code.strip().upper())

# ================= NanoDet 결과 파싱 =================
def extract_boxes(nanodet_result) -> List[Tuple[float, int, int, int, int]]:
    # NanoDet 모델 출력 결과에서 바운딩 박스목록 추출.
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
    # 수평 투영을 이용한 주요 텍스트 밴드 추출.
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
    # 이미지 좌우의 불필요한 공백 제거
    col = (img < 128).mean(axis=0)
    nz = np.where(col > 0.02)[0]
    if nz.size == 0: return img
    return img[:, nz[0]:nz[-1]+1]

def split_letter_digits_cut(token: np.ndarray) -> Optional[int]:
    # 문자와 숫자 부분을 분리하기 위한 최적의 수직 분할 지점 탐색
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
    # 수직 투영을 이용해 공백 기준으로 텍스트 밴드를 여러 토큰으로 분할
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
    # 공백 1개 → 오른쪽을 문자/숫자로 추가 분할
    g0 = (int(gaps[0][0]), int(gaps[0][-1]))
    cmid = (g0[0] + g0[1]) // 2
    left = trim_lr(bin_band[:, :cmid]); right = trim_lr(bin_band[:, cmid:])
    cut = split_letter_digits_cut(right)
    if cut is None: return None
    return [left, trim_lr(right[:, :cut]), trim_lr(right[:, cut:])]

# ================= RapidOCR 래퍼 =================
class RapidRec:
    # RapidOCR 모델을 감싸 텍스트 인식을 수행하는 래퍼(Wrapper) 클래스
    def __init__(self):
        # lite 모델 기본, 영어 dict. 필요 시 모델 경로 인자로 바꿀 수 있음
        self.ocr = RapidOCR(rec_score_threshold=0.1)  # 낮게 잡고 후단에서 정규식/보팅
    def _run(self, img: np.ndarray, allow: str) -> Tuple[str, float]:
        # use_det=False로 "한 덩어리" 인식
        res, _ = self.ocr(img, use_det=False, use_cls=True)
        txt, score = "", 0.0
        try:
            if isinstance(res, list) and len(res) > 0:
                item = res[0]
                if isinstance(item, (list, tuple)):
                    # 형태 1: ['TEXT', 0.98]
                    if len(item) == 2 and isinstance(item[0], str):
                        txt, score = item[0], float(item[1])
                    # 형태 2: [[box], 'TEXT', 0.98]
                    elif len(item) >= 3 and isinstance(item[1], str):
                        txt, score = item[1], float(item[2])
        except Exception:
            pass
        # 화이트리스트 정리
        txt = "".join(ch for ch in txt.upper() if ch in allow)
        return txt, float(score)
    def digits(self, img: np.ndarray) -> str:
        t, _ = self._run(img, "0123456789")
        return t
    def letters(self, img: np.ndarray) -> str:
        t, _ = self._run(img, "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        return t
    def any_en(self, img: np.ndarray) -> str:
        t, _ = self._run(img, ALLOWLIST)
        return t

rapid = RapidRec()  # 전역 하나로 재사용(워밍업됨)

# 왼쪽 3자리 분할(강화) → RapidOCR로 글자별 인식
def split_three_chars(bin_img: np.ndarray) -> Optional[List[np.ndarray]]:
    # 이미지에 포함된 세 개의 글자를 각각의 이미지로 분할
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
    # 운송장 번호의 첫 세 자리 숫자를 엄격한 기준으로 인식
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
    # 분할 기반 OCR을 수행하여 운송장 번호 인식
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

        # 왼쪽 3자리: RapidOCR 글자별
        left_digits = read_left_digits_strict(left)
        strict_ok = left_digits is not None and len(left_digits) >= 3
        if not strict_ok:
            # 토큰 단위 보정
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
            return cand, bool(strict_ok)

    #전체 한 번 읽고 정규화
    merged = rapid.any_en(head)
    s = canon(merged)
    m = re.search(r"(\d{3})\s*([A-Z])\s*(\d{2})", s)
    if m:
        cand = f"{m.group(1)} {m.group(2)}{m.group(3)}"
        if CODE_REGEX.fullmatch(cand):
            if debug: print(f"[DEBUG] fallback(rapid) -> {cand}")
            return cand, False
    return None, False

def resize_for_detect(frame: np.ndarray, det_w: int):
    # 객체 탐지를 수행하기 전 프레임 크기 조정.
    h, w = frame.shape[:2]
    if w <= det_w: return frame, 1.0
    scale = det_w / float(w)
    small = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return small, scale

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=r"C:\Workspace\Robosort\RoboSort\nanodet_waybill.yml")
    parser.add_argument("--model",  default=r"C:\Workspace\Robosort\RoboSort\nanodet\workspace\waybill\model_last.ckpt")
    parser.add_argument("--camid", type=int, default=0)
    parser.add_argument("--conf",  type=float, default=0.35)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--det_every", type=int, default=DETECT_EVERY)
    parser.add_argument("--det_width", type=int, default=DETECT_WIDTH)
    parser.add_argument("--debug_ocr", action="store_true")
    parser.add_argument("--save_rois", action="store_true")
    parser.add_argument("--vote_min", type=int, default=DEFAULT_VOTE_MIN)
    args = parser.parse_args()

    VOTE_MIN = args.vote_min

    # NanoDet
    load_config(cfg, args.config)
    logger = Logger(-1, use_tensorboard=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = Predictor(cfg, args.model, logger, device=device)

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
    save_dir = os.path.join(script_dir, "roi_debug")
    if args.save_rois and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] camera read failed."); break

            # 1) 탐지
            need_detect = (frame_idx % max(1, args.det_every) == 0) or (best_box is None)
            vis = frame
            if need_detect:
                det_img, scale = resize_for_detect(frame, args.det_width)
                meta, res = predictor.inference(det_img)
                boxes = extract_boxes(res[0])
                chosen = None
                for score, x1, y1, x2, y2 in boxes:
                    if score < args.conf: continue
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
                if args.save_rois and roi.size>0 and frame_idx%10==0:
                    cv2.imwrite(os.path.join(save_dir, f"roi_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"), roi)

            # 3) OCR 제출(ROI 변동 없어도 표결 미완이면 계속)
            now = time.time()
            need_more_votes = len(vote_buf) < VOTE_MIN
            if roi is not None and (now - last_ocr_t) >= OCR_MIN_INTERVAL:
                if roi_changed(prev_roi, roi) or need_more_votes:
                    if pending is None or pending.done():
                        def worker(img, dbg):
                            return ocr_code_segmented(img, dbg)
                        pending = executor.submit(worker, roi.copy(), args.debug_ocr)
                        last_ocr_t = now
                        prev_roi = roi.copy()

            # 4) 결과 수거 + 즉시/연속/다수결
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

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    sys.exit(main())
