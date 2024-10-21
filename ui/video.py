import cv2 # type: ignore
import numpy as np
from config import load_calibration_data

class VideoStream:
    """Class for reading video frames from a video capture object."""
    
    def __init__(self, camera_index, fps=15, desired_width=640, desired_height=480, undistort=False):
        """Initialize the video stream with the given frame dimensions and camera index."""
        self.camera_index = camera_index
        self.vc = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        self.vc.set(cv2.CAP_PROP_FPS, fps)
        self.vc.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.vc.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        
        self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    
        self.frame_height = desired_height
        self.frame_width = desired_width
        
        self.undistort = undistort
        
        self.img = np.empty((self.frame_height, self.frame_width), dtype=np.uint32)
        self.view = self.img.view(dtype=np.uint8).reshape((self.frame_height, self.frame_width, 4))[::-1, ::]
        
        # Load camera calibration data
        try:
            self.camera_matrix, self.dist_coeffs, _, _ = load_calibration_data('config/new_camera_calibration.csv')
            self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, 
                                                                      self.dist_coeffs, 
                                                                      (self.frame_width, self.frame_height),
                                                                      1, # alpha
                                                                      (self.frame_width, self.frame_height))
            print(f"Camera matrix calculated, shape: {self.camera_matrix.shape}.")

        
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
            
            if self.undistort:
                print("Undistorting frame.")
                frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)

            self.view[:, :, 0] = frame[:, :, 2]
            self.view[:, :, 2] = frame[:, :, 0]
            self.view[:, :, 1] = frame[:, :, 1]
            self.view[:, :, 3] = 255 
            
    def take_snapshot(self):
        """Take a snapshot from the video stream."""
        print("Taking snapshot.")
        self.start()   
        self.get_frame()
        self.get_frame()
        self.stop()
        print("Snapshot taken.")
        
    def save_snapshot(self, filename):
        """Save the snapshot to a file."""
        cv2.imwrite(filename, self.view)
        print("Snapshot saved.")
    