import cv2
import sounddevice as sd # type: ignore
import acoular as ac # type: ignore
from bokeh.plotting import curdoc # type: ignore
from ui import Dashboard, VideoStream
from processing import ModelProcessor
from config import ConfigUMA, usb_camera_index, uma16_index

ac.config.global_caching = 'none'

# Initial configurations
VIDEO_SCALE_FACTOR = 1.0
video_index = usb_camera_index() 
mic_index = uma16_index()

# Initialize video capture
vc = cv2.VideoCapture(video_index)
frame_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH) * VIDEO_SCALE_FACTOR)
frame_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT) * VIDEO_SCALE_FACTOR)

# Initialize video stream and model processor
config = ConfigUMA()
video_stream = VideoStream(frame_width, frame_height, vc)
model_processor = ModelProcessor(frame_width, frame_height, config, mic_index)

# Create the UI layout using the Dashboard class
dashboard = Dashboard(video_stream, model_processor, config)

# Add the layout to the document
doc = curdoc()
doc.add_root(dashboard.get_layout())
