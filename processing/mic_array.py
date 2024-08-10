import sounddevice as sd # type: ignore
from pathlib import Path # type: ignore
#import tensorflow as tf # type: ignore
import acoular as ac # type: ignore
import numpy as np
from modelsdfg.transformer.config import ConfigBase # type: ignore
import matplotlib.pyplot as plt
import time
from config import ConfigUMA, uma16_index


config = ConfigUMA()
device_index = uma16_index()


def calc_csm(data):
    csm = ac.CSM(data=data, block_size=128, window='Hanning')
    return csm.csm[0, :, :]


sd.default.device = device_index 
source = ac.SoundDeviceSamplesGenerator(device=device_index, numchannels=16)

t = 1
ref_mic_index = config.datasets[0].pipeline.args['ref_mic_index']
source.numsamples = int(t * source.sample_freq)

print(source.numsamples)