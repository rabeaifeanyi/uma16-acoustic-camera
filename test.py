from data_processing import Processor  # Annahme: Processor-Klasse ist im data_processing Modul verfügbar
from config import ConfigUMA, uma16_index  # Annahme: Diese Konfigurationen sind vorhanden
import time
import acoular as ac #type:ignore

# Initialize configurations and Processor
config_uma = ConfigUMA()
mic_index = uma16_index()  # Platzhalter für deine Mikrofon-Index-Konfiguration
model_dir = "/home/rabea/Documents/Bachelorarbeit/models/EigmodeTransformer_learning_rate0.00025_epochs500_2024-04-10_19-09"
model_config_path = model_dir + "/config.toml"
ckpt_path = model_dir + '/ckpt/best_ckpt/0441-0.83.keras'
results_filename = "test_results"

ac.config.global_caching = 'none'

# Processor-Instanz definieren
processor = Processor(
    config_uma,
    mic_index,
    model_config_path,
    results_filename,
    ckpt_path,
    save_csv=False,
    save_h5=False
)


# processor.start_beamforming()

# time.sleep(3)

# #processor.stop_beamforming()

#time.sleep(2)

processor.start_model()

time.sleep(5)

processor.stop_model()