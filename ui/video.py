import cv2
import numpy as np


class VideoStream:
    def __init__(self, frame_width, frame_height, vc):
        self.vc = vc
        self.frame_width = frame_width
        self.frame_height = frame_height

    def get_frame(self):
        rval, frame = self.vc.read()
        if rval:
            # Resizing the frame to fit in the desired dimensions
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