import os
import cv2
import time
import serial
import numpy as np
from flask import Flask, Response, request, jsonify
from threading import Thread, Lock
from picamera2 import Picamera2
from libcamera import controls
import requests

app = Flask(__name__)

# ===========================================================
# Arduino Serial
# ===========================================================
try:
    ser = serial.Serial("/dev/ttyACM0", 9600, timeout=1)
    print("[INFO] Arduino connected")
    time.sleep(2)
except Exception as e:
    print("[WARN] Arduino not connected:", e)
    ser = None

serial_lock = Lock()

# -----------------------------------------
# -----------------------------------------
PC_IP = "10.112.249.124"
PC_COMPLETE_URL = f"http://{PC_IP}:6000/complete"


def send_to_arduino(line: bytes):
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
    except Exception:
        print("[WARN] Arduino write failed")
        ser = None
        return False


# ===========================================================
# ===========================================================
def arduino_listener():
    global ser
    if ser is None:
        print("[WARN] Arduino not connected -> listener disabled")
        return

    print("[ARDUINO] Listener started")

    while True:
        try:
            raw = ser.readline().decode(errors="ignore").strip()
            if not raw:
                continue

            print("[ARDUINO-RX]", raw)

            if raw == "DONE":
                print("[EVENT] DONE received -> notifying PC")
                try:
                    requests.post(PC_COMPLETE_URL, timeout=0.5)
                except Exception as e:
                    print("[WARN] Failed to notify PC:", e)

        except Exception as e:
            print("[ARDUINO] Listener exception:", e)
            time.sleep(0.5)


Thread(target=arduino_listener, daemon=True).start()

# ===========================================================
# Picamera2 (Main Cam)
# ===========================================================
picam2 = Picamera2()
cfg = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(cfg)
picam2.start()

try:
    picam2.set_controls({
        "AfMode": controls.AfModeEnum.Continuous,
        "AfTrigger": controls.AfTriggerEnum.Start,
    })
except Exception:
    print("[WARN] Autofocus not supported")


def gen_main_cam():
    while True:
        frame = picam2.capture_array()
        ok, jpeg = cv2.imencode(".jpg", frame)
        if ok:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )


@app.route("/cam")
def cam():
    return Response(gen_main_cam(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ===========================================================
# USB Webcam (Unload Cam)
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
usb_thread_running = True


def usb_loop():
    global usb_cap, usb_frame

    if USB_DEVICE is None:
        print("[USB] No USB device")
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
                print("[USB] Open failed, retry")
                time.sleep(1)
                continue

        ok, frame = usb_cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        with usb_lock:
            usb_frame = cv2.flip(frame, 1)

        time.sleep(1 / 30.0)

Thread(target=usb_loop, daemon=True).start()


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

            time.sleep(1 / 20.0)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ===========================================================
# API
# ===========================================================
@app.route("/send_code", methods=["POST"])
def send_code():
    data = request.json if request.is_json else {}
    code = data.get("code")

    try:
        zone = int(code)
    except Exception:
        return jsonify({"status": "error"}), 400

    print("[Pi] WAYBILL ZONE =", zone)
    send_to_arduino(f"{zone}\n".encode())

    return jsonify({"status": "ok", "sent": zone})


@app.route("/marker", methods=["POST"])
def marker():
    data = request.json if request.is_json else {}
    num = data.get("num")

    try:
        num = int(num)
    except Exception:
        return jsonify({"status": "error"}), 400

    print("[Pi] MARKER =", num)
    send_to_arduino(f"MARKER {num}\n".encode())

    return jsonify({"status": "ok", "sent": num})


# ===========================================================
# MAIN
# ===========================================================
if __name__ == "__main__":
    print("===================================")
    print("         Pi Server Running         ")
    print("===================================")

    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        usb_thread_running = False
        time.sleep(0.3)

        try:
            picam2.stop()
        except Exception:
            pass

        if ser:
            try:
                ser.close()
            except Exception:
                pass