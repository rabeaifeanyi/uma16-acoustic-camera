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

def calculate_view_range(Z, ratio=(4, 3), dx=None, dy=None, dz=None, alpha_x=None, alpha_y=None):
    if alpha_x is not None and alpha_y is not None:
        pass
    
    elif dx is not None and dz is not None:
        alpha_x = 2 * np.arctan(dx / (2 * dz))
        alpha_y = 2 * np.arctan((ratio[1] * dx) / (2 * ratio[0] * dz))
        
    elif dy is not None and dz is not None:
        alpha_y = 2 * np.arctan(dy / (2 * dz))
        alpha_x = 2 * np.arctan((ratio[0] * dy) / (2 * ratio[1] * dz))
    
    elif alpha_x is None or alpha_y is None:
        raise ValueError("Not enough information provided to calculate view range.")
        
    xmax = Z * np.tan(alpha_x / 2)
    xmin = -xmax
    ymax = Z * np.tan(alpha_y / 2)
    ymin = -ymax
    
    print(f"View range: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
    
    return xmin, xmax, ymin, ymax


def calculate_alphas(Z, ratio=(4, 3), dx=None, dy=None, dz=None):    
    if dx is not None and dz is not None:
        alpha_x = 2 * np.arctan(dx / (2 * dz))
        alpha_y = 2 * np.arctan((ratio[1] * dx) / (2 * ratio[0] * dz))
        
    elif dy is not None and dz is not None:
        alpha_y = 2 * np.arctan(dy / (2 * dz))
        alpha_x = 2 * np.arctan((ratio[0] * dy) / (2 * ratio[1] * dz))

    return alpha_x, alpha_y