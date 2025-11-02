import cv2
import numpy as np
import time
import json
import argparse
import serial, serial.tools.list_ports as list_ports
import sys
# pigpio를 라즈베리파이에서만 쓰도록 안전하게 감싸기
try:
    import pigpio
    PIGPIO_AVAILABLE = True
except Exception:
    PIGPIO_AVAILABLE = False
from collections import deque

# ================== [GUI-MPL] Matplotlib Viewer ==================
class MatplotlibViewer:
    """OpenCV imshow 대신 Matplotlib로 프레임 표시"""
    def __init__(self, title="RoboSort - Nav View"):
        import matplotlib
        try:
            matplotlib.use("TkAgg")  # 가능하면 TkAgg 백엔드 사용
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
        plt.ion()
        plt.show()

    def _on_key(self, event):
        if event.key in ("q", "escape"):
            self.stop = True

    def update(self, bgr_img: np.ndarray):
        # OpenCV BGR -> Matplotlib RGB
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        if self.im is None:
            self.im = self.ax.imshow(rgb)
        else:
            self.im.set_data(rgb)
        self.plt.pause(0.001)

    def close(self):
        try:
            self.fig.canvas.mpl_disconnect(self.cid)
        except Exception:
            pass
        self.plt.ioff()
        self.plt.close(self.fig)
# ================================================================

# ---------- 유틸 ----------
def find_serial_port(prefer=None):
    if prefer: return prefer
    ports = list_ports.comports()
    for p in ports:
        name = (p.device or "") + " " + (p.description or "")
        if any(k in name for k in ("Arduino", "wchusb", "ttyACM", "ttyUSB", "COM")):
            return p.device
    return ports[0].device if ports else None

# ---------- 아두이노 직렬 ----------
class MotorSerial:
    def __init__(self, port=None, baud=115200):
        self.port_name = find_serial_port(port)
        self.ser = None
        if self.port_name:
            try:
                self.ser = serial.Serial(self.port_name, baud, timeout=0.05)
                time.sleep(1.5)
                print(f'[SER] opened {self.port_name}@{baud}')
            except Exception as e:
                print(f'[SER] open failed: {e}')
        else:
            print('[SER] no port found (dry-run mode)')

    def send_vel(self, l, r):
        l = int(np.clip(l, -255, 255)); r = int(np.clip(r, -255, 255))
        line = f"VEL {l} {r}\n"
        if self.ser: self.ser.write(line.encode())
        print(f"[TX] {line.strip()}")

    def stop(self):
        line = "STOP\n"
        if self.ser: self.ser.write(line.encode())
        print(f"[TX] {line.strip()}")

    def read_line(self):
        if not self.ser: return None
        try:
            s = self.ser.readline().decode("utf-8", errors="ignore").strip()
            return s if s else None
        except:
            return None

# ---------- 서보 ----------
class PanServo:
    # 서보 각도: -90 ~ +90 deg → 펄스 500~2500us 맵핑 (모델 따라 조정)
    def __init__(self, gpio_pin=18, us_min=500, us_max=2500):
        # Windows이거나 pigpio 미탑재면 더미 모드로 자동 전환
        self._dummy = (not PIGPIO_AVAILABLE) or (sys.platform.startswith("win"))
        self._angle = 0.0
        self.pin = gpio_pin
        self.us_min, self.us_max = us_min, us_max

        if self._dummy:
            print("[WARN] pigpio unavailable or non-Pi platform → Dummy PanServo (no physical servo).")
            # 더미 모드: 실제 GPIO 제어 없이 논리 각도만 유지
            return

        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio not connected. Run: sudo pigpiod")
        self.set_angle(0)

    def angle_to_us(self, deg):
        deg = np.clip(deg, -90, 90)
        t = (deg + 90) / 180.0
        return int(self.us_min + t * (self.us_max - self.us_min))

    def set_angle(self, deg):
        self._angle = float(np.clip(deg, -90, 90))
        if not self._dummy:
            self.pi.set_servo_pulsewidth(self.pin, self.angle_to_us(self._angle))

    @property
    def angle(self):
        return self._angle

    def close(self):
        if not self._dummy:
            try:
                self.pi.set_servo_pulsewidth(self.pin, 0)
                self.pi.stop()
            except:
                pass

# ---------- 비전: ArUco ----------
class ArucoFinder:
    def __init__(self, cam_id=0, fov_deg=62.0, dict_name=cv2.aruco.DICT_4X4_50,
                 hflip=False,           
                 use_eq=True
                 ):
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.fov = float(fov_deg)
        self.hflip = bool(hflip)     
        self.use_eq = bool(use_eq)   

        self.dict = cv2.aruco.getPredefinedDictionary(dict_name)

        self.param = cv2.aruco.DetectorParameters()
        self.param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.param.adaptiveThreshWinSizeMin = 3
        self.param.adaptiveThreshWinSizeMax = 31
        self.param.adaptiveThreshWinSizeStep = 4
        self.param.minMarkerPerimeterRate = 0.02
        self.param.maxMarkerPerimeterRate = 4.0
        self.param.minDistanceToBorder = 3
        # 필요 시: self.param.adaptiveThreshConstant = 7

        self.detector = cv2.aruco.ArucoDetector(self.dict, self.param)

    def _preproc(self, frame):
        if not self.use_eq:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        if self.hflip:                       
            frame = cv2.flip(frame, 1)
        frame = self._preproc(frame)         
        return True, frame

    def find_ids(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)
        return corners, (ids.flatten() if ids is not None else None)

    def bearing_from_center(self, frame_w, center_x):
        # 이미지 중앙 대비 픽셀 오프셋 → 각도(도)
        offset = (center_x - frame_w/2) / (frame_w/2)  # -1..+1
        return offset * (self.fov / 2.0)

    def close(self):
        try: self.cap.release()
        except: pass

# ---------- 상태 머신 ----------
class NavFSM:
    def __init__(self, ms: MotorSerial, servo: PanServo, vf: ArucoFinder, target_id: int, show=False, debug=False):  # [ADDED debug]
        self.ms, self.servo, self.vf = ms, servo, vf
        self.target_id = int(target_id)
        self.show = show
        self.debug = debug  
        self.state = "SCAN"   # SCAN -> TRACK_ALIGN -> APPROACH -> DROP -> RETREAT -> DONE
        self.scan_dir = +1
        self.scan_deg = 0
        self.last_seen = time.time()
        self.seen_once = False
        self.align_ok_since = None
        self.front_near = False

        # 파라미터(필요시 조정)
        self.scan_step = 12.0        # 스캔 시 서보 각도 스텝(도)
        self.scan_wait = 0.12        # 스캔 위치에서 settle 시간(초)
        self.k_turn = 3.0            # 회전 게인 (deg -> PWM)
        self.k_corr = 2.0            # 전진 중 헤딩 보정 게인
        self.base_fwd = 120          # 접근시 기본 전진 PWM
        self.turn_max = 180          # 회전 PWM 최대치
        self.align_tol_deg = 5.0     # 정렬 완료 판정(절대 방위각)
        self.align_hold = 0.4        # 정렬 유지 시간(초)
        self.approach_timeout = 12.0 # 접근 최대 시간(안전)
        self.retreat_time = 1.0      # 후진 시간
        self.done = False

        # ================== [GUI-MPL] ==================
        self.viewer = MatplotlibViewer("RoboSort - Navigation") if self.show else None
        # =================================================

    def _limit(self, v, lim): return int(np.clip(v, -lim, lim))

    def tick(self):
        # 아두이노 이벤트 수신(전방 근접)
        line = self.ms.read_line()
        if line:
            print(f'[SER<-] {line}')
            if 'FRONT_NEAR' in line:
                self.front_near = True

        ok, frame = self.vf.read()
        if not ok:
            self.ms.stop()
            return frame

        vis = frame.copy()  # 오버레이용
        h, w = frame.shape[:2]
        corners, ids = self.vf.find_ids(frame)

        if self.debug:  
            print("detected IDs:", None if ids is None else ids.tolist())

        cx = None
        if ids is not None:
            for i, mid in enumerate(ids):
                if mid == self.target_id:
                    # 해당 마커 중심 x
                    pts = corners[i][0]  # 4x2
                    cx = float(np.mean(pts[:,0]))
                    self.last_seen = time.time()
                    self.seen_once = True
                    break

        # 디버그 오버레이
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        cv2.putText(vis, f"STATE:{self.state}", (10,30), 0, 1, (0,255,0), 2)
        cv2.putText(vis, f"SERVO:{self.servo.angle:+.1f} deg", (10,60), 0, 1, (0,255,255), 2)

        # 상태별 로직
        if self.state == "SCAN":
            # 좌↔우 훑으며 ID 찾기
            if cx is not None:
                self.state = "TRACK_ALIGN"
                self.align_ok_since = None
            else:
                # 스텝 스캔
                if abs(self.servo.angle - self.scan_deg) > 1.0:
                    # 각도 도달 대기
                    pass
                else:
                    # 다음 스텝으로
                    self.scan_deg += self.scan_dir * self.scan_step
                    if self.scan_deg > 90:  self.scan_deg, self.scan_dir = 90, -1
                    if self.scan_deg < -90: self.scan_deg, self.scan_dir = -90, +1
                    self.servo.set_angle(self.scan_deg)
                    time.sleep(self.scan_wait)
                self.ms.stop()

        elif self.state == "TRACK_ALIGN":
            if cx is None:
                # 일시 미검출 → 잠깐 스캔 재개
                if time.time() - self.last_seen > 1.0:
                    self.state = "SCAN"
                    self.scan_dir = +1 if self.servo.angle < 0 else -1
                self.ms.stop()
            else:
                # 서보가 마커 중앙을 따라가게(픽셀 오차 -> 서보 보정)
                bearing_err = self.vf.bearing_from_center(w, cx)  # 화면 중심 기준 각도오차(도)
                target_servo = np.clip(self.servo.angle + bearing_err, -90, 90)
                self.servo.set_angle(target_servo)

                # 로봇은 "서보 각도 ≈ 0"이 되도록 제자리 회전
                yaw_err = self.servo.angle
                turn = self._limit(self.k_turn * yaw_err, self.turn_max)
                self.ms.send_vel(-turn, +turn)

                # 정렬 판정
                if abs(yaw_err) <= self.align_tol_deg:
                    if self.align_ok_since is None:
                        self.align_ok_since = time.time()
                    elif time.time() - self.align_ok_since >= self.align_hold:
                        self.state = "APPROACH"
                        self.approach_since = time.time()
                else:
                    self.align_ok_since = None

        elif self.state == "APPROACH":
            if self.front_near:
                self.ms.stop()
                self.state = "DROP"
            elif time.time() - self.approach_since > self.approach_timeout:
                print("[WARN] approach timeout → STOP")
                self.ms.stop()
                self.state = "DROP"
            else:
                if cx is not None:
                    # 계속 중앙에 오도록 서보 보정
                    bearing_err = self.vf.bearing_from_center(w, cx)
                    self.servo.set_angle(np.clip(self.servo.angle + bearing_err, -90, 90))
                # 헤딩 보정
                yaw_err = self.servo.angle
                corr = self._limit(self.k_corr * yaw_err, 100)
                l = self._limit(self.base_fwd - corr, 255)
                r = self._limit(self.base_fwd + corr, 255)
                self.ms.send_vel(l, r)

        elif self.state == "DROP":
            print("[ACTION] DROP (placeholder)")
            self.ms.stop()
            time.sleep(1.0)  # 투하 대기 (나중에 경사판 제어 추가)
            self.state = "RETREAT"
            self.retreat_until = time.time() + self.retreat_time

        elif self.state == "RETREAT":
            self.ms.send_vel(-140, -140)
            if time.time() >= self.retreat_until:
                self.ms.stop()
                self.state = "DONE"
                self.done = True

        # ================== [GUI-MPL] 화면 갱신 ==================
        if self.viewer is not None:
            self.viewer.update(vis)
            if self.viewer.stop:
                self.done = True
        # ========================================================

        return vis

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camid", type=int, default=0)
    ap.add_argument("--serial_port", type=str, default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--servo_pin", type=int, default=18)
    ap.add_argument("--map", type=str, default="aruco_map.json")
    ap.add_argument("--zone", type=int, required=True, help="OCR로 결정된 목표 구역(예: 600)")
    ap.add_argument("--show", action="store_true")

    ap.add_argument("--hflip", action="store_true", help="웹캠 좌우반전 보정")
    ap.add_argument("--no_eq", action="store_true", help="명암 평활화 끄기")
    ap.add_argument("--debug", action="store_true", help="탐지된 ID 디버그 출력")

    args = ap.parse_args()

    with open(args.map, "r", encoding="utf-8") as f:
        m = json.load(f)
    fov = float(m.get("camera_fov_deg", 62.0))
    zone_to_aruco = {int(k): int(v) for k,v in m.get("zone_to_aruco", {}).items()}
    if args.zone not in zone_to_aruco:
        print(f"[ERR] zone {args.zone} not in aruco map"); return
    target_id = zone_to_aruco[args.zone]
    print(f"[MAP] zone {args.zone} -> ArUco ID {target_id}, FOV={fov}°")

    ms = MotorSerial(args.serial_port, args.baud)
    servo = PanServo(args.servo_pin)  # pigpio 없으면 자동 dummy
    vf = ArucoFinder(args.camid, fov_deg=fov, hflip=args.hflip, use_eq=not args.no_eq)  

    try:
        fsm = NavFSM(ms, servo, vf, target_id, show=args.show, debug=args.debug)  
        while not fsm.done:
            fsm.tick()
            time.sleep(0.01)
    finally:
        ms.stop()
        servo.close()
        vf.close()
        fsm.close()

if __name__ == "__main__":
    main()
