import cv2
import cv2.aruco as aruco

# 저장될 경로
SAVE_DIR = "./markers"

# 사용할 딕셔너리 (4x4_1000 → ID 0~999 사용 가능)
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)

nums = [100, 200, 300, 400, 500, 600]

for n in nums:
    marker = aruco.drawMarker(aruco_dict, n, 400)  # 400px 정사각형
    cv2.imwrite(f"{SAVE_DIR}/{n}.png", marker)

print("완료: markers/*.png 생성됨")
