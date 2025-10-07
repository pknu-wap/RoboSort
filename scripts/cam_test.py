import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    cv2.imshow("Camera Test", frame)

    # ESC 누르면 종료
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
