import acoular as ac # type: ignore
from bokeh.plotting import curdoc # type: ignore
from ui import Dashboard, VideoStream
from processing import ModelProcessor
from config import ConfigUMA, uma16_index, calculate_alphas

ac.config.global_caching = 'none'

# Video configurations
VIDEO_SCALE_FACTOR = 1 # TODO überlegen, was hier schlau
UNDISTORT = False
DUMMY = False # Später herausnehmen
Z = 2 #m
DX, DZ = 143, 58 #m # TODO genauer Messen aber auch Alternativberechnung implementieren
alphas = calculate_alphas(Z, dx=DX, dz=DZ) # TODO Datenblatt finden und Winkel überprüfen

# Update rate configurations in ms
ESTIMATION_UPDATE_INTERVAL = 1000
CAMERA_UPDATE_INTERVAL = 100
STREAM_UPDATE_INTERVAL = 1000

# Model paths
model_dir = "/home/rabea/Documents/Bachelorarbeit/models/EigmodeTransformer_learning_rate0.00025_epochs500_2024-04-10_19-09"
model_config_path = model_dir + "/config.toml"
ckpt_path = model_dir + '/ckpt/best_ckpt/0441-0.83.keras'

video_index = 0
mic_index = uma16_index()
STARTFREQ = 5000

# Initialize video stream and model processor
config_uma = ConfigUMA()
video_stream = VideoStream(video_index, VIDEO_SCALE_FACTOR, UNDISTORT)
frame_width = video_stream.frame_width
frame_height = video_stream.frame_height

model_processor = ModelProcessor(
    config_uma, 
    mic_index,
    model_config_path, 
    ckpt_path,
    STARTFREQ)

# Create the UI layout using the Dashboard class
dashboard = Dashboard(
    video_stream, 
    model_processor, 
    config_uma, 
    ESTIMATION_UPDATE_INTERVAL, 
    CAMERA_UPDATE_INTERVAL, 
    STREAM_UPDATE_INTERVAL,
    alphas,
    dummy=DUMMY)

# Add the layout to the document
doc = curdoc()
doc.add_root(dashboard.get_layout())
