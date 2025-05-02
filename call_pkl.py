import pickle

calibration_file_path = r'/home/addinedu/Documents/GitHub/calibration/camera_calibration.pkl'

def load_calibration_data(pkl_file_path):
    """Load calibration data from pickle file."""
    with open(pkl_file_path, 'rb') as f:
        calibration_data = pickle.load(f)
    return calibration_data