import cv2
import numpy as np
import glob
import os
import pickle

# Base path 설정
base_path = r'/home/addinedu/Documents/GitHub/calibration'
plkfile_save_path = os.path.join(base_path, 'checkerboards')
os.makedirs(plkfile_save_path, exist_ok=True)

save_dir = r'/home/addinedu/Documents/GitHub/calibration/pkl'

def get_unique_filename(base_name='camera_calibration', ext='pkl', save_dir='/home/addinedu/Documents/GitHub/calibration/pkl'):
    i = 1
    filename = os.path.join(save_dir, f"{base_name}.{ext}")
    
    if os.path.exists(filename):
        print(f"[INFO] Calibration file already exists: {filename}")
        re_make_file = input("Do you want to make another file? (yes/no): ").strip().lower()
        
        if re_make_file == 'yes':
            while os.path.exists(filename):
                filename = os.path.join(save_dir, f"{base_name}_{i}.{ext}")
                i += 1
        elif re_make_file == 'no':
            print("Overwriting existing file...")
        else:
            print("Invalid input. Overwriting existing file by default.")

    return filename

def calibrate_camera():
    CHECKERBOARD = (9, 6)
    SQUARE_SIZE = 25.0

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints = []

    images = glob.glob(os.path.join(plkfile_save_path, '*.png'))
    print(f"[INFO] {len(images)} images loaded.")

    if not images:
        print("[ERROR] No checkerboard images found.")
        return None, None, None, None, None, None, None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('Checkerboard Detection', img)
            cv2.waitKey(100)

    cv2.destroyAllWindows()

    if not objpoints or not imgpoints:
        print("[ERROR] Calibration failed: No valid checkerboard detected.")
        return None, None, None, None, None, None, None

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    print("\n[CALIBRATION RESULT]")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)

    unique_file = get_unique_filename('camera_calibration', 'pkl', save_dir)

    calibration_data = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'rvecs': rvecs,
        'tvecs': tvecs
    }

    with open(unique_file, 'wb') as f:
        pickle.dump(calibration_data, f)
    print(f"[Saved] Calibration data saved to {unique_file}")

    return calibration_data, ret, objpoints, rvecs, imgpoints, images, tvecs

def live_video_correction(calibration_data):
    mtx = calibration_data["camera_matrix"]
    dist = calibration_data["dist_coeffs"]

    cap = cv2.VideoCapture(2)
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read first frame.")
        cap.release()
        return

    h, w = frame.shape[:2]
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    crop_x, crop_y = 60, 120
    crop_w, crop_h = 800, 270

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        cropped_original = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        cropped_dst = dst[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

        combined = np.hstack((cropped_original, cropped_dst))

        scale = 2.0
        h, w = combined.shape[:2]
        combined_resized = cv2.resize(combined, (int(w*scale), int(h*scale)))

        cv2.imshow('Original (Left) | Corrected (Right)', combined_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def error_check(ret, objpoints, rvecs, tvecs, mtx, dist, imgpoints, images):
    print(f"[전체 리프로젝션 에러 (RMS)]: {ret:.4f}")

    print("\n[이미지별 리프로젝션 에러]")
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        print(f"Image {i+1}: {error:.4f} pixels")

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)

        combined = np.hstack((img, undistorted_img))
        cv2.imshow(f"Original vs Undistorted - Image {idx+1}", combined)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibration_data, ret, objpoints, rvecs, imgpoints, images, tvecs = calibrate_camera()

    if calibration_data:
        live_video_correction(calibration_data)
        error_check(ret, objpoints, rvecs, tvecs,
                    calibration_data["camera_matrix"], calibration_data["dist_coeffs"],
                    imgpoints, images)
