from pathlib import Path
import acoular as ac # type: ignore
import tensorflow as tf # type: ignore
from modelsdfg.transformer.config import ConfigBase # type: ignore
import numpy as np
import sounddevice as sd # type: ignore
from config import ConfigUMA, uma16_index



def slow_estimator(source):

    model_dir = "/home/rabea/Documents/Bachelorarbeit/models/EigmodeTransformer_learning_rate0.00025_epochs500_2024-04-10_19-09"

    config_path = model_dir + "/config.toml"
    ckpt_path = model_dir + '/ckpt/best_ckpt/0441-0.83.keras'
    #ckpt_path = model_dir + '/ckpt/final_model.keras'
    micgeom_path = Path(ac.__file__).parent / 'xml' / 'minidsp_uma16.xml'
    
    mics = ac.MicGeom(from_file=micgeom_path)

    # load the config object
    config = ConfigBase.from_toml(config_path)
    pipeline = config.datasets[1].pipeline.create_instance()
    print(f"Model was trained on 1/3rd octave band frequencies: {config.datasets[0].training.f}")

    config.datasets[1].validation.cache=False # do not cache the dataset

    t = config.datasets[0].config_args['signal_length']
    print(f"Signal length: {t}")

    ref_mic_index = config.datasets[0].pipeline.args['ref_mic_index']
    print(f"Reference microphone index: {ref_mic_index}")

    model = tf.keras.models.load_model(ckpt_path)

    #fft_data = ac.FFTSpectra(time_data=source, block_size=128, window='Hanning')
    
    freq_data = ac.PowerSpectra(
        time_data=source, 
        block_size=128, 
        window='Hanning'
        )
    
    f_ind = np.searchsorted(freq_data.fftfreq(), 800)
    print("f index:", f_ind)
    csm = freq_data.csm[f_ind] 
    
    lower_freq = 0
    upper_freq = 10000
    freq_indices = np.where((freq_data.fftfreq() >= lower_freq) & (freq_data.fftfreq() <= upper_freq))[0]
    summed_csm = np.zeros_like(freq_data.csm[freq_indices[0]])
    
    
    for f_ind in freq_indices:
        summed_csm += freq_data.csm[f_ind]
        
    csm = csm / 0.0016**2 # später kalibrieren

    csm_norm = csm[ref_mic_index,ref_mic_index]
    csm = csm / csm_norm
    
    eigmode = model.preprocessing(csm[np.newaxis]).numpy()
    print("Eigenmode shape:", eigmode.shape)
    
    strength_pred, loc_pred, noise_pred = model.predict(eigmode)
    strength_pred = strength_pred.squeeze()
    strength_pred *= np.real(csm_norm)
    
    loc_pred = pipeline.recover_loc(loc_pred.squeeze(), aperture=mics.aperture)

    print("prediction strength:", strength_pred)
    print("predicion loc:", loc_pred)
    
    
    return strength_pred, loc_pred, noise_pred

device_index = uma16_index()
t=1
sd.default.device = device_index 
source = ac.SoundDeviceSamplesGenerator(device=device_index, numchannels=16) # object, responsible for recording the time data 
source.numsamples = int(t * source.sample_freq)

strength_pred, loc_pred, noise_pred = slow_estimator(source)

print("strength_pred:", strength_pred)
print("loc_pred:", loc_pred)
print("noise_pred:", noise_pred)

