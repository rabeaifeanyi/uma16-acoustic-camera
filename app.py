import acoular as ac # type: ignore
import datetime
import os
from bokeh.plotting import curdoc # type: ignore
from ui import Dashboard, VideoStream
from data_processing import Processor
from config import ConfigUMA, uma16_index, calculate_alphas

ac.config.global_caching = 'none' # type: ignore

# Video configurations
UNDISTORT = False
Z = 3 #m
MIN_DISTANCE = 1 #m
THRESHOLD = 0.001 
DESIRED_WIDTH = 640
DESIRED_HEIGHT = 480
FPS = 20
SCALE_FACTOR = 1.5

DX, DZ = 143, 58 #m # TODO genauer Messen aber auch Alternativberechnung implementieren
alphas = calculate_alphas(Z, dx=DX, dz=DZ) # TODO Datenblatt finden und Winkel überprüfen

# Configuration for saving results
CSV = False
H5 = True

# Update rate configurations in ms
ESTIMATION_UPDATE_INTERVAL = 100
BEAMFORMING_UPDATE_INTERVAL = 1000
CAMERA_UPDATE_INTERVAL = 100
STREAM_UPDATE_INTERVAL = 1000

# Model paths
model_dir = '/home/rabea/Documents/Bachelorarbeit/models/EigmodeTransformer_learning_rate0.00025_weight_decay1e-06_epochs500_2024-10-16_16-51'
model_config_path = model_dir + '/config.toml'
ckpt_path = model_dir + '/ckpt/best_ckpt/0187-1.09.keras'

# Folder for results
results_folder = 'messungen'

# check if folder exists
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Initialize video stream and model processor
video_index = 2
device = uma16_index()

# Initialize video stream and model processor
config_uma = ConfigUMA()

# Switch to turn camera on
#video_stream = VideoStream(video_index, undistort=UNDISTORT, fps=FPS, desired_width=DESIRED_WIDTH, desired_height=DESIRED_HEIGHT)
video_stream = False

processor = Processor(
    device,
    results_folder,
    ckpt_path,
    CSV,
    H5)

dashboard = Dashboard(
    video_stream, 
    processor, 
    config_uma, 
    ESTIMATION_UPDATE_INTERVAL, 
    BEAMFORMING_UPDATE_INTERVAL,
    CAMERA_UPDATE_INTERVAL, 
    STREAM_UPDATE_INTERVAL,
    THRESHOLD,
    alphas,
    SCALE_FACTOR,
    Z,
    MIN_DISTANCE)

doc = curdoc()
doc.add_root(dashboard.get_layout())
