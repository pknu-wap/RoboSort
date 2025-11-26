import os
import cv2
import time
import serial
import numpy as np
from flask import Flask, Response, request, jsonify
from threading import Thread, Lock
from picamera2 import Picamera2
from libcamera import controls

# ===========================================================
#                Arduino Serial
# ===========================================================
try:
    ser = serial.Serial("/dev/ttyACM0", 9600, timeout=1)
    print("[INFO] Arduino connected")
    time.sleep(2)
except Exception as e:
    print("[WARN] Arduino not connected:", e)
    ser = None

serial_lock = Lock()

def send_to_arduino(line: bytes) -> bool:
    global ser
    if ser is None:
        print("[ARDUINO-DRY]", line)
        return False
    try:
        with serial_lock:
            ser.write(line)
            ser.flush()
        print("[ARDUINO-TX]", line)
        return True
    except Exception as e:
        print("[WARN] Arduino write failed:", e)
        ser = None
        return False

# ===========================================================
#                   FSM States
# ===========================================================
FSM_IDLE = "IDLE"
FSM_MOVING = "MOVING_TO_TARGET"
FSM_UNLOADING = "UNLOADING"
FSM_RETURNING = "RETURNING_HOME"

fsm_state = FSM_IDLE
is_busy = False
current_target_zone = None
arduino_reader_running = True

# ===========================================================
#                Arduino Reader Thread
# ===========================================================
def arduino_reader_loop():
    global ser, fsm_state, is_busy, current_target_zone

    print("[INFO] Arduino reader started")

    while arduino_reader_running:
        if ser is None:
            time.sleep(0.2)
            continue

        try:
            line = ser.readline()
            if not line:
                continue

            s = line.decode(errors="ignore").strip()
            if not s:
                continue

            print("[ARDUINO-RX]", s)

            if s == "UNLOAD_DONE":
                fsm_state = FSM_RETURNING
                send_to_arduino(b"BACK\n")

            elif s == "DONE":
                is_busy = False
                fsm_state = FSM_IDLE
                current_target_zone = None

        except Exception as e:
            print("[WARN] Arduino read failed:", e)
            ser = None
            time.sleep(0.5)

Thread(target=arduino_reader_loop, daemon=True).start()

# ===========================================================
#               Picamera2 Main Camera
# ===========================================================
picam2 = Picamera2()
cfg = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(cfg)
picam2.start()

picam2.set_controls({
    "AfMode": controls.AfModeEnum.Continuous,
    "AfTrigger": controls.AfTriggerEnum.Start,
})

def gen_main_cam():
    while True:
        frame = picam2.capture_array()
        ok, jpeg = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )

# ===========================================================
#            USB Webcam (Unload Camera) Auto Detect
# ===========================================================
def find_usb_device():
    for dev in ["/dev/video0", "/dev/video1", "/dev/video2", "/dev/video4"]:
        cap = cv2.VideoCapture(dev)
        if cap.isOpened():
            cap.release()
            print("[USB] Found device:", dev)
            return dev
    print("[USB] No USB camera found")
    return None

USB_DEVICE = find_usb_device()

usb_cap = None
usb_frame = None
usb_lock = Lock()
usb_thread_running = False

def usb_loop():
    global usb_cap, usb_frame, usb_thread_running

    if USB_DEVICE is None:
        print("[USB] No USB device detected -> cannot start loop")
        return

    print("[USB] usb_loop started")

    while usb_thread_running:
        if usb_cap is None:
            cap = cv2.VideoCapture(USB_DEVICE, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                usb_cap = cap
                print("[USB] Webcam opened:", USB_DEVICE)
            else:
                print("[USB] open failed, retry")
                time.sleep(1)
                continue

        ok, frame = usb_cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        with usb_lock:
            usb_frame = cv2.flip(frame, 1)

        time.sleep(1.0 / 30.0)

def start_usb_thread():
    global usb_thread_running
    if usb_thread_running:
        return
    usb_thread_running = True
    Thread(target=usb_loop, daemon=True).start()
    print("[USB] Started USB capture thread")

def stop_usb_thread():
    global usb_thread_running, usb_cap
    usb_thread_running = False
    if usb_cap:
        try:
            usb_cap.release()
        except:
            pass
        usb_cap = None
    print("[USB] Stopped USB capture thread")

# ===========================================================
#                Flask Web Server
# ===========================================================
app = Flask(__name__)

@app.route("/cam")
def cam():
    return Response(gen_main_cam(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/unload_cam")
def unload_cam():
    def gen():
        while True:
            with usb_lock:
                frame = usb_frame

            if frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                ok, jpeg = cv2.imencode(".jpg", blank)
            else:
                ok, jpeg = cv2.imencode(".jpg", frame)

            if ok:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n"
                )

            time.sleep(1.0 / 20.0)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ===========================================================
#                    API: PC -> Pi
# ===========================================================
@app.route("/send_code", methods=["POST"])
def send_code():
    global fsm_state, is_busy, current_target_zone

    data = request.json if request.is_json else {}
    code = data.get("code")

    if is_busy:
        return jsonify({"status": "busy"})

    try:
        zone = int(code)
    except:
        return jsonify({"status": "error"}), 400

    current_target_zone = zone
    fsm_state = FSM_MOVING
    is_busy = True

    send_to_arduino(b"FWD\n")

    return jsonify({"status": "ok", "target": zone})

@app.route("/unload_number", methods=["POST"])
def unload_number():
    global fsm_state, current_target_zone

    data = request.json if request.is_json else {}
    num = data.get("num")

    try:
        num = int(num)
    except:
        return jsonify({"status": "error"}), 400

    print("[Pi] Marker:", num, " target:", current_target_zone, " state:", fsm_state)

    if fsm_state == FSM_MOVING and num == current_target_zone:
        send_to_arduino(b"STOP\n")

        if num < 400:
            send_to_arduino(b"UNLOAD_L\n")
        else:
            send_to_arduino(b"UNLOAD_R\n")

        fsm_state = FSM_UNLOADING

    elif fsm_state == FSM_RETURNING and num == 100:
        send_to_arduino(b"STOP\n")
        send_to_arduino(b"HOME\n")

    return jsonify({"status": "ok"})

# ===========================================================
#                     MAIN
# ===========================================================
if __name__ == "__main__":
    print("===================================")
    print("     Pi Simple FSM Server          ")
    print("===================================")

    start_usb_thread()

    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        usb_thread_running = False
        arduino_reader_running = False
        time.sleep(0.3)

        try:
            picam2.stop()
        except:
            pass

        if ser:
            try:
                ser.close()
            except:
                pass
