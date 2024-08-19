import acoular as ac # type: ignore
from bokeh.plotting import curdoc # type: ignore
from ui import Dashboard, VideoStream
from processing import ModelProcessor
from config import ConfigUMA, uma16_index, calculate_alphas
import numpy as np

ac.config.global_caching = 'none'

# Video configurations
VIDEO_SCALE_FACTOR = 1
UNDISTORT = False
Z = 2 
# TODO genauer Messen und nochmal scharf nachdenken, ob das wirklich so einfach ist
DX, DZ = 143, 58
alphas = calculate_alphas(Z, dx=DX, dz=DZ)

# Update rate configurations
ESTIMATION_UPDATE_INTERVAL = 1000 #ms
CAMERA_UPDATE_INTERVAL = 100
STREAM_UPDATE_INTERVAL = 1000

# Model paths
model_dir = "/home/rabea/Documents/Bachelorarbeit/models/EigmodeTransformer_learning_rate0.00025_epochs500_2024-04-10_19-09"
model_config_path = model_dir + "/config.toml"
ckpt_path = model_dir + '/ckpt/best_ckpt/0441-0.83.keras'

video_index = 2 #usb_camera_index() 
mic_index = uma16_index()

# Initialize video stream and model processor
config_uma = ConfigUMA()
video_stream = VideoStream(video_index, VIDEO_SCALE_FACTOR, UNDISTORT)
frame_width = video_stream.frame_width
frame_height = video_stream.frame_height
model_processor = ModelProcessor(frame_width, 
                                 frame_height, 
                                 config_uma, 
                                 mic_index, 
                                 model_dir, 
                                 model_config_path, 
                                 ckpt_path)

# Create the UI layout using the Dashboard class
dashboard = Dashboard(video_stream, 
                      model_processor, 
                      config_uma, 
                      ESTIMATION_UPDATE_INTERVAL, 
                      CAMERA_UPDATE_INTERVAL, 
                      STREAM_UPDATE_INTERVAL,
                      alphas)

# Add the layout to the document
doc = curdoc()
doc.add_root(dashboard.get_layout())
