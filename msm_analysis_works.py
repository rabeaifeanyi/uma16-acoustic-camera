#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import acoular as ac 
from pathlib import Path
import matplotlib.pyplot as plt

FREQ = 4000 

#%% Beamforming

filename = "messungen/2024-10-17_14-53-06_model_time_data.h5"
#freq_data=ac.PowerSpectra(time_data=ac.MaskedTimeSamples(name=Path(__file__).parent / filename),window='Hanning', overlap='50%', block_size=128)

freq_data=ac.PowerSpectra(time_data=ac.TimeSamples(name=Path(__file__).parent / filename),window='Hanning', overlap='50%', block_size=128)

mics=ac.MicGeom(from_file=Path(ac.__file__).parent / 'xml' / 'minidsp_uma-16_mirrored.xml')
    
grid=ac.RectGrid(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, z=1.15, increment=0.05)



#%%

import tensorflow as tf 
import numpy as np
import matplotlib



model_name = "EigmodeTransformer_learning_rate0.00025_weight_decay1e-06_epochs500_2024-10-16_16-51"
model_dir = Path(f"/home/rabea/Documents/Bachelorarbeit/models/{model_name}")
config_path = model_dir / 'config.toml'
ckpt_path = model_dir / 'ckpt' / 'best_ckpt'
ckpt_files = ckpt_path.glob('*.keras')
ckpt_name = sorted(ckpt_files, key=lambda x: int(x.stem.split('-')[0]))[-1].name
ckpt_path = model_dir / 'ckpt' / 'best_ckpt'/ ckpt_name
print(f"Using checkpoint: {ckpt_path}")

# load the machine learning model
model = tf.keras.models.load_model(ckpt_path)

csm = freq_data.csm[:,:,:]
# normalize the CSM
csm_norm = csm[:,0,0]
csm = csm / csm_norm[:, np.newaxis, np.newaxis]
fftfreq = freq_data.fftfreq()
idx = np.argmin(np.abs(fftfreq - FREQ))

csm=csm[idx][np.newaxis]
eigmode = model.preprocessing(csm).numpy()

# # make prediction
strength_pred, loc_pred, noise_pred = model.predict(eigmode)
loc_pred -= 0.5
loc_pred *= 2.0
strength_pred = strength_pred.squeeze()
loc_pred = loc_pred.squeeze()
strength_pred *= np.real(csm_norm)[idx]
print(loc_pred, strength_pred)

# plot the results
fig, ax = plt.subplots()
ax.set_title("Transformer model prediction")
vmax = ac.L_p(strength_pred).max()
vmin = vmax - 20 # 20 dB dynamic range
norm = matplotlib.colors.Normalize(vmax=vmax,vmin=vmin)
color = matplotlib.pyplot.get_cmap('hot_r')(norm(ac.L_p(strength_pred)))
im = ax.imshow(np.zeros((64,64)), origin='lower', extent=(-1.5, 1.5, -1.5, 1.5), cmap='hot_r', vmax=vmax, vmin=vmin)
c = plt.colorbar(im, ax=ax, label=r'$L_{p}$ [dB]',pad=0.02, shrink=1,)
ax.set_xlabel(r'$x$ [m]')
ax.set_ylabel(r'$y$ [m]')
# plot predicted locations
for s in range(len(loc_pred[0])):
    if ac.L_p(strength_pred)[s] > 0:
        ax.plot(loc_pred[0,s], loc_pred[1,s], 'o', markersize=2.5, alpha=0.8, markeredgewidth=0, color=color[s])
plt.show()

