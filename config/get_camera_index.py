import cv2

def usb_camera_index():
    """Get the index of the USB camera.
    """
    usb_camera_found = False

    indices_to_check = [3, 2, 1, 0]
    for index in indices_to_check:
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if cap.isOpened():
            if index in [2, 3]: 
                usb_camera_found = True
            cap.release()

    return 2 if usb_camera_found else 0