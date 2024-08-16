import cv2
import numpy as np
from config import load_calibration_data

class VideoStream:
    """Class for reading video frames from a video capture object."""
    
    def __init__(self, frame_width, frame_height, vc, camera_index):
        """Initialize the video stream with the given frame dimensions and video capture object."""
        self.vc = vc
        self.camera_index = camera_index  # Store the camera index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.camera_matrix, self.dist_coeffs, _, _ = load_calibration_data('config/camera_calibration.csv')
        self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (self.frame_width, self.frame_height), 1, (self.frame_width, self.frame_height))

    def start(self):
        """Start the video capture if it's not already running."""
        if not self.vc.isOpened():
            self.vc.open(self.camera_index)  # Use the correct camera index

    def stop(self):
        """Stop the video capture."""
        if self.vc.isOpened():
            self.vc.release()

    def get_frame(self, undistort=False):
        """Read a frame from the video capture object and return it as an RGBA image."""
        rval, frame = self.vc.read()
        
        if rval:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            if undistort:
                frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)
            
            # Convert the frame to RGBA
            img = np.empty((self.frame_height, self.frame_width), dtype=np.uint32)
            view = img.view(dtype=np.uint8).reshape((self.frame_height, self.frame_width, 4))[::-1, ::-1]
            view[:, :, 0] = frame[:, :, 2]
            view[:, :, 2] = frame[:, :, 0]
            view[:, :, 1] = frame[:, :, 1]
            view[:, :, 3] = 255  # Set alpha channel to fully visible
            return img
        
        return None
