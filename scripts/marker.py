import cv2
import numpy as np

dict_name = cv2.aruco.DICT_4X4_50
aruco_dict = cv2.aruco.getPredefinedDictionary(dict_name)

def save_marker(marker_id, side_px=800, pad_ratio=0.20, path=None, border_bits=1):
    marker = np.full((side_px, side_px), 255, dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, marker_id, side_px, marker, border_bits)
    pad = int(side_px * pad_ratio)
    marker = cv2.copyMakeBorder(marker, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=255)
    path = path or f"aruco_{marker_id}.png"
    cv2.imwrite(path, marker)
    print("saved:", path)

for zid, mid in {0:10,100:11,200:12,300:13,400:14,500:15,600:16,700:17,800:18,900:19}.items():
    save_marker(mid, side_px=800, pad_ratio=0.20, path=f"zone{zid}_id{mid}.png")

