import cv2 # type: ignore
import numpy as np
from config import load_calibration_data

class VideoStream:
    """Class for reading video frames from a video capture object."""
    
    def __init__(self, camera_index, undistort=False):
        """Initialize the video stream with the given frame dimensions and camera index."""
        self.camera_index = camera_index
        self.vc = cv2.VideoCapture(camera_index)
        self.frame_width = int(self.vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.undistort = undistort
        
        # Load camera calibration data
        try:
            self.camera_matrix, self.dist_coeffs, _, _ = load_calibration_data('config/camera_calibration.csv')
            self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, 
                                                                      self.dist_coeffs, 
                                                                      (self.frame_width, self.frame_height),
                                                                      1, # alpha
                                                                      (self.frame_width, self.frame_height))
        except FileNotFoundError:
            self.camera_matrix = None
            self.dist_coeffs = None
            self.new_camera_matrix = None
        
    def start(self):
        """Start the video capture if it's not already running."""
        if not self.vc.isOpened():
            self.vc.open(self.camera_index)

    def stop(self):
        """Stop the video capture."""
        if self.vc.isOpened():
            self.vc.release()

    def get_frame(self):
        """Read a frame from the video capture object and return it as an RGBA image."""
        rval, frame = self.vc.read()

        if rval:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            
            if self.undistort and self.camera_matrix:
                frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)

            img = np.empty((self.frame_height, self.frame_width), dtype=np.uint32)
            view = img.view(dtype=np.uint8).reshape((self.frame_height, self.frame_width, 4))[::-1, ::-1]
            view[:, :, 0] = frame[:, :, 2]
            view[:, :, 2] = frame[:, :, 0]
            view[:, :, 1] = frame[:, :, 1]
            view[:, :, 3] = 255 
   
            return img
        
        return None

