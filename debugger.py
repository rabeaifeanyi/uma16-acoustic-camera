import numpy as np
import tensorflow as tf #type: ignore
from modelsdfg.transformer.config import ConfigBase #type: ignore
from pathlib import Path
import acoular as ac #type: ignore
from config import ConfigUMA

    
def create_data():
    sfreq = 51200
    duration = 1
    nsamples = duration * sfreq
    
    uma = ConfigUMA()
    h5savefile = Path('three_sources.h5')

    m = uma.create_mics()
    n1 = ac.WNoiseGenerator(sample_freq=sfreq, numsamples=nsamples, seed=1)
    n2 = ac.WNoiseGenerator(sample_freq=sfreq, numsamples=nsamples, seed=2, rms=0.7)
    n3 = ac.WNoiseGenerator(sample_freq=sfreq, numsamples=nsamples, seed=3, rms=0.5)
    p1 = ac.PointSource(signal=n1, mics=m, loc=(-0.1, -0.1, 0.3))
    p2 = ac.PointSource(signal=n2, mics=m, loc=(0.15, 0, 0.3))
    p3 = ac.PointSource(signal=n3, mics=m, loc=(0, 0.1, 0.3))
    p = ac.Mixer(source=p1, sources=[p2, p3])
    wh5 = ac.WriteH5(source=p, name=h5savefile)
    wh5.save()

def read_data():
    uma = ConfigUMA()
    mg = uma.create_mics()
    ts = ac.TimeSamples(name="three_sources.h5")
    ps = ac.PowerSpectra(time_data=ts, block_size=128, window='Hanning')
    return ps
    
    
class ModelDebugger:
    def __init__(self, model_config_path, ckpt_path):
        # Modell-Setup
        self._setup_model(model_config_path, ckpt_path)

    def _setup_model(self, model_config_path, ckpt_path):
        # Laden der Modellkonfiguration
        model_config = ConfigBase.from_toml(model_config_path)
        self.pipeline = model_config.datasets[1].pipeline.create_instance()
        self.ref_mic_index = model_config.datasets[0].pipeline.args['ref_mic_index']
        # Laden des Modells
        self.model = tf.keras.models.load_model(ckpt_path)
        print("Modell erfolgreich geladen.")

    def preprocess_csm(self, csm):
        # Beispielhafte Vorverarbeitung des CSM
        csm_norm = csm[self.ref_mic_index, self.ref_mic_index]
        csm = csm / csm_norm
        eigmode = self.model.preprocessing(csm[np.newaxis]).numpy()
        return eigmode

    def predict(self, eigmode):
        # Modellvorhersage
        strength_pred, loc_pred, noise_pred = self.model.predict(eigmode, verbose=0)
        return strength_pred, loc_pred, noise_pred

if __name__ == "__main__":
    
    #Pfade zum Modell und zur Konfiguration (anpassen)
    
    model_dir = "/home/rabea/Documents/Bachelorarbeit/models/EigmodeTransformer_learning_rate0.00025_epochs500_2024-04-10_19-09"
    model_config_path = model_dir + "/config.toml"
    ckpt_path = model_dir + '/ckpt/best_ckpt/0441-0.83.keras'
    # Initialisierung des Debuggers
    debugger = ModelDebugger(model_config_path, ckpt_path)

    # Beispielhaftes CSM erzeugen (16x16 Matrix)
    csm_gen = read_data()
    csm = csm_gen.calc_csm()
    csm = csm[26]

    # Vorverarbeitung
    eigmode = debugger.preprocess_csm(csm)

    # Prediction durchführen
    strength_pred, loc_pred, noise_pred = debugger.predict(eigmode)
    strength_pred = strength_pred.squeeze()
    strength_pred *= np.real(csm_norm)
    loc_pred = pipeline.recover_loc(loc_pred.squeeze(), aperture=mics.aperture)
    

    # Ergebnisse ausgeben
    print("Stärke Vorhersage:", strength_pred)
    print("Positionsvorhersage:", loc_pred)
