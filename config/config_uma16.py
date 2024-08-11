import acoular as ac # type: ignore
from acoupipe.datasets.synthetic import DatasetSyntheticConfig, DatasetSynthetic # type: ignore
from pathlib import Path
from scipy.stats import uniform # type: ignore
import acoupipe.sampler as sp # type: ignore
from traits.api import Dict # type: ignore
import matplotlib.pyplot as plt
import sounddevice as sd # type: ignore


# Messbereich in m
YMIN_MEASUREMENT = -1.5
YMAX_MEASUREMENT = 1.5
XMIN_MEASUREMENT = -1.5
XMAX_MEASUREMENT = 1.5
Z = 2.0

INCREMENT = 3/63
MAXNSOURCES = 9

# UMA aperture: 0.178 m

class ConfigUMA(DatasetSyntheticConfig):
    """Configuration for the UMA-16 microphone array.

    Based on an example from the acoupipe documentation.
    https://adku1173.github.io/acoupipe/contents/jupyter/modify.html
    """
    fft_params = Dict({
                    'block_size' : 256,
                    'overlap' : '50%',
                    'window' : 'Hanning',
                    'precision' : 'complex64'},
                desc='FFT parameters')

    def create_mics(self):
        """Create the microphone array.
        """
        uma_file = Path(ac.__file__).parent / 'xml' / 'minidsp_uma-16.xml'
        return ac.MicGeom(from_file=uma_file)

    def create_grid(self):
        """Create a grid for the sources.
        """
        return ac.RectGrid(y_min=YMIN_MEASUREMENT, y_max=YMAX_MEASUREMENT, x_min=XMIN_MEASUREMENT, x_max=XMAX_MEASUREMENT, z=Z, increment=INCREMENT)

    def create_location_sampler(self):
        """Create a location sampler for the sources.
        """
        location_sampler = sp.LocationSampler(
            random_var = (uniform(-1.5,3),uniform(-1.5,3),uniform(2.0,0)),
            nsources = MAXNSOURCES,
            )
        
        # Default:
        # location_sampler = sp.LocationSampler(
        #     random_var = (norm(0,0.1688*ap),norm(0,0.1688*ap),norm(z,0)), #2d also passt z so
        #     x_bounds = (-0.5*ap,0.5*ap),
        #     y_bounds = (-0.5*ap,0.5*ap),
        #     z_bounds = (0.5*ap,0.5*ap),
        #     nsources = self.max_nsources,
        #     mindist = 0.1*ap,)
        
        if self.snap_to_grid:
            location_sampler.grid = self.source_grid
        
        return location_sampler
    
    def mic_positions(self):
        """Get the positions of the microphones.
        """
        return self.mics.mpos
    

def uma16_index():
    """Get the index of the UMA-16 microphone array.
    """
    devices = sd.query_devices()

    for index, device in enumerate(devices):
        if "nanoSHARC micArray16 UAC2.0" in device['name']:
            device_index = index
            print(f"\nUMA-16 device: {device['name']} at index {device_index}\n")
            break
    else:
        print("Could not find the UMA-16 device. Defaulting to the first device")
        device_index = 0 # Default to the first device
        
    return device_index
