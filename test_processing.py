from data_processing import Processor  
from config import ConfigUMA, uma16_index 
import time
import acoular as ac #type:ignore

config_uma = ConfigUMA()
mic_index = uma16_index()
model_dir = "/home/rabea/Documents/Bachelorarbeit/models/EigmodeTransformer_learning_rate0.00025_epochs100_2024-10-09_09-03"
model_config_path = model_dir + "/config.toml"
ckpt_path = model_dir + '/ckpt/best_ckpt/0078-1.06.keras'
results_filename = "test_results"
 
ac.config.global_caching = 'none' # type: ignore

processor = Processor(
    config_uma,
    mic_index,
    model_config_path,
    results_filename,
    ckpt_path,
    save_csv=False,
    save_h5=False
)

processor.start_beamforming()
time.sleep(5)
processor.stop_beamforming()

# processor.start_model()
# time.sleep(5)
# processor.stop_model()

# processor.start_beamforming()
# time.sleep(5)
# processor.stop_beamforming()
