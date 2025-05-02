import cv2
import numpy as np
import os
import glob
import pickle

def calibrate_camera():
    CHECKERBOARD = (6, 9)  
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []
    imgpoints = [] 
    used_images = []  #체스보드를 성공적으로 찾은 이미지만 저장

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    images = glob.glob('checkerboards/*.png')

    if not images:
        raise FileNotFoundError("[ERROR] checkerboards 폴더에 .png 이미지가 없습니다.")

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"[WARNING] 이미지를 읽지 못했습니다: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            used_images.append(fname)  #성공한 경우에만 저장

            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
        else:
            print(f"[WARNING] 체스보드를 찾지 못했습니다: {fname}")

    cv2.destroyAllWindows()

    if len(objpoints) == 0 or len(imgpoints) == 0:
        raise ValueError("[ERROR] 캘리브레이션에 사용할 유효한 체스보드 이미지가 없습니다.")

    # 🛠 캘리브레이션 수행
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("Camera matrix:\n", mtx)
    print("\nDistortion coefficients:\n", dist)

    # 🛠 에러 체크 호출 (used_images 사용)
    error_check(ret, objpoints, rvecs, tvecs, mtx, dist, imgpoints, used_images)

    # 결과 저장
    calibration_data = {
        'camera_matrix': mtx,
        'dist_coeffs': dist,
        'rvecs': rvecs,
        'tvecs': tvecs
    }
    
    pkl_save_path = r'/home/addinedu/Documents/GitHub/calibration/pkl'
    
    with open(pkl_save_path, 'wb') as f:
        pickle.dump(calibration_data, f)

    return calibration_data


def error_check(ret, objpoints, rvecs, tvecs, mtx, dist, imgpoints, images):
    print(f"\n[전체 리프로젝션 에러 (RMS)]: {ret:.4f}")

    print("\n[이미지별 리프로젝션 에러]")
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        print(f"Image {i+1}: {error:.4f} pixels")

    print("\n[왜곡 제거 이미지 확인]")
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            continue
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)

        combined = np.hstack((img, undistorted_img))
        cv2.imshow(f"Original vs Undistorted - Image {idx+1}", combined)
        cv2.waitKey(500)  # 500ms 보여주기

    cv2.destroyAllWindows()

def live_video_correction(calibration_data):
    mtx = calibration_data['camera_matrix']
    dist = calibration_data['dist_coeffs']
    cap = cv2.VideoCapture(2)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        if all(v > 0 for v in [x, y, w, h]):
            dst = dst[y:y+h, x:x+w]
        # original = cv2.resize(frame, (640, 480))
        corrected = cv2.resize(dst, (640, 480))
        # corrected_ = np.hstack(corrected)
        # combined = np.hstack((original, corrected))
        # cv2.imshow('Original | Corrected', combined)
        cv2.imshow('Corrected', corrected)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    pkl_path = r'/home/addinedu/Documents/GitHub/calibration/pkl/camera_calibration.pkl'
    if os.path.exists(pkl_path):
        print("Loading existing calibration data...")
        with open(pkl_path, 'rb') as f:
            calibration_data = pickle.load(f)

        print("skip error check since existing calibration data has been loaded")

    else:
        print("Performing new camera calibration...")
        calibration_data = calibrate_camera()

    print("Starting live video correction...")
    live_video_correction(calibration_data)
