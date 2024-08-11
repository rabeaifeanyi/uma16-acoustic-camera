import cv2
import numpy as np

class VideoStream:
    """Class for reading video frames from a video capture object.
    """
    def __init__(self, frame_width, frame_height, vc):
        """Initialize the video stream with the given frame dimensions and video capture object.
        
        Args:
            frame_width (int): The width of the frame.
            frame_height (int): The height of the frame.
            vc (cv2.VideoCapture): The video capture object.
        """
        self.vc = vc
        self.frame_width = frame_width
        self.frame_height = frame_height

    def get_frame(self):
        """Read a frame from the video capture object and return it as an RGBA image.
        """
        rval, frame = self.vc.read()
        
        if rval:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            
            # Converting the frame to RGBA
            img = np.empty((self.frame_height, self.frame_width), dtype=np.uint32)
            view = img.view(dtype=np.uint8).reshape((self.frame_height, self.frame_width, 4))[::-1, ::-1]
            view[:, :, 0] = frame[:, :, 2]
            view[:, :, 2] = frame[:, :, 0]
            view[:, :, 1] = frame[:, :, 1]
            view[:, :, 3] = 255  # Set alpha channel to fully visible
            return img
        
        return None
