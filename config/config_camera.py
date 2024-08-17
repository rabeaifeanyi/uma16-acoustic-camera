import cv2
import numpy as np
import csv

def load_calibration_data(csv_file):
    """Load camera calibration data from a CSV file."""
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
        
        # Camera Matrix (3x3)
        camera_matrix = np.array(data[0][1:], dtype=np.float32).reshape((3, 3))
        
        # Distortion Coefficients
        dist_coeffs = np.array(data[1][1:], dtype=np.float32)
        
        # Rotation and Translation Vectors
        r_vecs = []
        t_vecs = []
        for i in range(2, len(data), 2):
            r_vec = np.array(data[i][1:], dtype=np.float32).reshape((3, 1))
            t_vec = np.array(data[i+1][1:], dtype=np.float32).reshape((3, 1))
            r_vecs.append(r_vec)
            t_vecs.append(t_vec)
    
    return camera_matrix, dist_coeffs, r_vecs, t_vecs

def usb_camera_index():
    # TODO: Implement this method, this does not work, indices are chaning
    """Get the index of the USB camera. 
    This method is not universal and may need to be adjusted for different systems.
    """
    usb_camera_found = False

    indices_to_check = [0, 2]
    for index in indices_to_check:
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if cap.isOpened():
            if index == 2: 
                usb_camera_found = True
            cap.release()
            
    return 2 