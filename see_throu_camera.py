import cv2

cap = cv2.VideoCapture(3)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("USB Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

