import acoular as ac # type: ignore
from processing import ModelProcessorNew
from config import ConfigUMA, uma16_index, calculate_alphas
import time

ac.config.global_caching = 'none'

# Video configurations
VIDEO_SCALE_FACTOR = 1
UNDISTORT = False
Z = 2 #m
DX, DZ = 143, 58 #m # TODO genauer Messen
alphas = calculate_alphas(Z, dx=DX, dz=DZ)

# Update rate configurations
ESTIMATION_UPDATE_INTERVAL = 1000 #ms
CAMERA_UPDATE_INTERVAL = 100
STREAM_UPDATE_INTERVAL = 1000

# Model paths
model_dir = "/home/rabea/Documents/Bachelorarbeit/models/EigmodeTransformer_learning_rate0.00025_epochs500_2024-04-10_19-09"
model_config_path = model_dir + "/config.toml"
ckpt_path = model_dir + '/ckpt/best_ckpt/0441-0.83.keras'

video_index = 0
mic_index = uma16_index()

# Initialize video stream and model processor
config_uma = ConfigUMA()
model_processor = ModelProcessorNew(config_uma, 
                                 mic_index,
                                 model_config_path, 
                                 ckpt_path)

try:
    model_processor.start_model()
    input("Press Enter to stop the model...")
finally:
    model_processor.stop_model()
    print("Model has been stopped.")