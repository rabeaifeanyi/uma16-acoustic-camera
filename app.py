import acoular as ac # type: ignore
import datetime
from bokeh.plotting import curdoc # type: ignore
from ui import Dashboard, VideoStream
from data_processing import Processor
from config import ConfigUMA, uma16_index, calculate_alphas

ac.config.global_caching = 'none' # type: ignore

# Video configurations
UNDISTORT = False
Z = 3 #m
MIN_DISTANCE = 1 #m
THRESHOLD = 0 #dB

DX, DZ = 143, 58 #m # TODO genauer Messen aber auch Alternativberechnung implementieren
alphas = calculate_alphas(Z, dx=DX, dz=DZ) # TODO Datenblatt finden und Winkel überprüfen

# Configuration for saving results
CSV = False
H5 = False

# Update rate configurations in ms
ESTIMATION_UPDATE_INTERVAL = 1000
BEAMFORMING_UPDATE_INTERVAL = 1000
CAMERA_UPDATE_INTERVAL = 100
STREAM_UPDATE_INTERVAL = 1000

# Model paths
model_dir = "/home/rabea/Documents/Bachelorarbeit/models/EigmodeTransformer_learning_rate0.00025_epochs100_2024-10-09_09-03"
model_config_path = model_dir + "/config.toml"
ckpt_path = model_dir + '/ckpt/best_ckpt/0078-1.06.keras'

# Filename for results
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
results_filename = f"results_{current_time}"

video_index = 0
mic_index = uma16_index()

# Initialize video stream and model processor
config_uma = ConfigUMA()
video_stream = VideoStream(video_index, undistort=UNDISTORT)
frame_width = video_stream.frame_width
frame_height = video_stream.frame_height

processor = Processor(
    config_uma, 
    mic_index,
    model_config_path, 
    results_filename,
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
    Z,
    MIN_DISTANCE)

doc = curdoc()
doc.add_root(dashboard.get_layout())
