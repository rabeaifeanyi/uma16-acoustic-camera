import cv2
import sounddevice as sd # type: ignore
import acoular as ac # type: ignore
from bokeh.plotting import curdoc # type: ignore
from ui import Dashboard, VideoStream
from processing import ModelProcessor
from config import ConfigUMA, usb_camera_index, uma16_index

ac.config.global_caching = 'none'

# Initial configurations
VIDEO_SCALE_FACTOR = 1
UNDISTORT = True

ESTIMATION_UPDATE_INTERVAL = 1000 #ms
CAMERA_UPDATE_INTERVAL = 100
STREAM_UPDATE_INTERVAL = 500

video_index = usb_camera_index() 
mic_index = uma16_index()

# Initialize video stream and model processor
config = ConfigUMA()
video_stream = VideoStream(video_index, VIDEO_SCALE_FACTOR, UNDISTORT)
frame_width = video_stream.frame_width
frame_height = video_stream.frame_height
model_processor = ModelProcessor(frame_width, frame_height, config, mic_index)

# Create the UI layout using the Dashboard class
dashboard = Dashboard(video_stream, 
                      model_processor, 
                      config, 
                      ESTIMATION_UPDATE_INTERVAL, 
                      CAMERA_UPDATE_INTERVAL, 
                      STREAM_UPDATE_INTERVAL)

# Add the layout to the document
doc = curdoc()
doc.add_root(dashboard.get_layout())
