import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import h5py # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
import sounddevice as sd # type: ignore
from pathlib import Path
import tensorflow as tf # type: ignore
import acoular as ac
import matplotlib.pyplot as plt # type: ignore
import matplotlib # type: ignore

from pathlib import Path


results_folder = 'messungen/'

result_list = os.listdir(results_folder)

print("Anzahl an Ergebnissen:", len(result_list)/2)

test_result = result_list[-2]
test_samples = result_list[-1]

print(test_samples, test_result)

result_filename_long = "messungen/2024-10-17_14-50-30_model_results.h5"
sample_filename_long = "messungen/2024-10-17_14-50-30_model_time_data.h5"

result_filename = "messungen/2024-10-17_14-53-06_model_results.h5"
sample_filename = "/home/rabea/Documents/Bachelorarbeit/uma16_acoustic_camera/messungen/2024-10-17_14-53-06_model_time_data.h5"

channel_index = 0

def recover_loc(loc, aperture, shift_loc=True, norm_loc=2):
        
    if shift_loc:
        if isinstance(shift_loc, float):
            loc = loc - shift_loc
        else:
            loc = loc - 0.5
    if norm_loc:
        if isinstance(norm_loc, float):
            loc = loc * norm_loc
        else:
            loc = loc * aperture
    return loc


model_name = "EigmodeTransformer_learning_rate0.00025_weight_decay1e-06_epochs500_2024-10-16_16-51"
model_dir = Path(f"/home/rabea/Documents/Bachelorarbeit/models/{model_name}")
config_path = model_dir / 'config.toml'
ckpt_path = model_dir / 'ckpt' / 'best_ckpt'
ckpt_files = ckpt_path.glob('*.keras')
ckpt_name = sorted(ckpt_files, key=lambda x: int(x.stem.split('-')[0]))[-1].name
ckpt_path = model_dir / 'ckpt' / 'best_ckpt'/ ckpt_name


# model_dir = '/home/rabea/Documents/Bachelorarbeit/models/EigmodeTransformer_learning_rate0.00025_weight_decay1e-06_epochs500_2024-10-16_16-51'
# model_config_path = model_dir + '/config.toml'
# ckpt_path = model_dir + '/ckpt/best_ckpt/0187-1.09.keras'
FREQ = 4000 #Hz

micgeom_path = Path(ac.__file__).parent / 'xml' / 'minidsp_uma-16_mirrored.xml'
mics = ac.MicGeom(from_file=micgeom_path)

ref_mic_index = 0

# load the machine learning model
model = tf.keras.models.load_model(ckpt_path)

devices = sd.query_devices()

for index, device in enumerate(devices):
    if "nanoSHARC micArray16 UAC2.0" in device['name']:
        device_index = index
        print(f"\nAusgewähltes Eingabegerät: {device['name']} bei Index {device_index}\n")
        break
else:
    print("Kein geeignetes Eingabegerät gefunden.")

sd.default.device = device_index 


source = ac.TimeSamples(name=sample_filename)
samplerate = sd.default.samplerate

# microphone geometry
mics = ac.MicGeom(from_file=micgeom_path)
csm_block_size = 128

# calculate the CSM from the incoming data
freq_data = ac.PowerSpectra(
    time_data=source, 
    block_size=csm_block_size, 
    window='Hanning'
    )

f_ind = np.argmin(np.abs(freq_data.fftfreq() - FREQ))
idx = f_ind

def preprocess_csm(csm):
    """ Preprocess the CSM data
    """
    neig = 8
    evls, evecs = np.linalg.eigh(csm)
    eigmode = evecs[..., -neig:] * evls[:, np.newaxis, -neig:]
    eigmode = np.stack([np.real(eigmode), np.imag(eigmode)], axis=3)
    eigmode = np.transpose(eigmode, [0, 2, 1, 3])
    input_shape = np.shape(eigmode)
    eigmode = np.reshape(eigmode, [-1, input_shape[1], input_shape[2]*input_shape[3]])
    return eigmode


csm = freq_data.csm[:,:,:]
# normalize the CSM
csm_norm = csm[:,0,0]
csm = csm / csm_norm[:, np.newaxis, np.newaxis]

csm=csm[idx][np.newaxis]

eigmode = preprocess_csm(csm)
# make prediction
strength_pred, loc_pred, noise_pred = model.predict(eigmode)
print(loc_pred)
strength_pred = strength_pred.squeeze()
print(np.real(csm_norm)[idx])
strength_pred *= np.real(csm_norm)[idx]
#loc_pred = recover_loc(loc_pred.squeeze(), aperture=mics.aperture, shift_loc=True, norm_loc=2) 

loc_pred -= 0.5
loc_pred *= 2.0
print(loc_pred)

# recover absolute location

# # plot the results
# fig, ax = plt.subplots()
# ax.set_title("Transformer model prediction")
# vmax = ac.L_p(strength_pred).max()
# vmin = vmax - 20 # 20 dB dynamic range
# norm = matplotlib.colors.Normalize(vmax=vmax,vmin=vmin)
# color = matplotlib.pyplot.get_cmap('hot_r')(norm(ac.L_p(strength_pred)))
# im = ax.imshow(np.zeros((64,64)), origin='lower', extent=(-1.5, 1.5, -1.5, 1.5), cmap='hot_r', vmax=vmax, vmin=vmin)
# c = plt.colorbar(im, ax=ax, label=r'$L_{p}$ [dB]',pad=0.02, shrink=1,)
# ax.set_xlabel(r'$x$ [m]')
# ax.set_ylabel(r'$y$ [m]')
# # plot microphone array as unfilled circles
# for m in mics.mpos.T:
#     ax.plot(m[0], m[1], 'ko', markersize=2., markeredgewidth=0.25, fillstyle='none')
# # plot predicted locations
# for s in range(loc_pred.shape[1]):
#     if ac.L_p(strength_pred)[s] > 0:
#         ax.plot(loc_pred[0,s], loc_pred[1,s], 'o', markersize=2.5, alpha=0.8, markeredgewidth=0, color=color[s])
