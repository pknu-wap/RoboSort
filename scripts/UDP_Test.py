import socket
import numpy as np
import cv2
import matplotlib.pyplot as plt

UDP_IP = "0.0.0.0"
UDP_PORT = 50010

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("[INFO] Waiting for frames...")

plt.ion()  # interactive mode on
fig, ax = plt.subplots()
img_handle = None

while True:
    data, addr = sock.recvfrom(65536)

    # decode JPEG
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if img_handle is None:
        img_handle = ax.imshow(frame_rgb)
    else:
        img_handle.set_data(frame_rgb)

    plt.pause(0.001)
