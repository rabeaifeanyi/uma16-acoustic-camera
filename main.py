import cv2
import sounddevice as sd # type: ignore
from bokeh.plotting import curdoc
from bokeh.layouts import column
from ui import *
from processing import *
from config import *


VIDEO_INDEX = usb_camera_index() # 2 for external camera if connected else 0 for webcam
MIC_INDEX = uma16_index() # index of internal Microphone if UMA16 is not connected
VIDEO_SCALE_FACTOR = 1.0 # TODO: better solution for scaling the video


# Open the video capture
vc = cv2.VideoCapture(VIDEO_INDEX)
frame_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH) * VIDEO_SCALE_FACTOR)
frame_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT) * VIDEO_SCALE_FACTOR)

# Initialize the video stream and model processor
config = ConfigUMA()
video_stream = VideoStream(frame_width, frame_height, vc)
model_processor = ModelProcessor(frame_width, frame_height, config)

# Get the microphone positions
mic_positions = config.mic_positions()

# Create the plot
fig, cds, mic_cds, cameraCDS = create_plot(frame_width, frame_height, mic_positions)


def update_camera_view():
    """Update the camera view with the latest frame.
    """
    img = video_stream.get_frame()
    if img is not None:
        cameraCDS.data['image_data'] = [img]

def update_estimations():
    """Update the estimations on the plot.
    """
    model_data = model_processor.get_uma16_dummy_data()
    update_plot(cds, model_data)
    

doc = curdoc()
doc.add_periodic_callback(update_estimations, 1000)
doc.add_periodic_callback(update_camera_view, 100)
doc.add_root(column(fig))
