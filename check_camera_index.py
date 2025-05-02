import cv2

def find_real_working_cameras(max_index=10):
    print("[INFO] Searching for real working cameras...")

    real_working_cameras = []

    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[FOUND] Camera truly working at index {index}")
                real_working_cameras.append(index)
            else:
                print(f"[BROKEN] Camera at index {index} cannot read frame.")
            cap.release()
        else:
            print(f"[EMPTY] No camera at index {index}")

    if not real_working_cameras:
        print("[WARN] No truly working cameras found.")

    return real_working_cameras

if __name__ == "__main__":
    available_cameras = find_real_working_cameras(max_index=10)
    print("\nReal working camera indices:", available_cameras)
