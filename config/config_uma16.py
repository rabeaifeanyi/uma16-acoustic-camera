import acoular as ac # type: ignore
from acoupipe.datasets.synthetic import DatasetSyntheticConfig, DatasetSynthetic # type: ignore
from pathlib import Path
from scipy.stats import uniform # type: ignore
import acoupipe.sampler as sp # type: ignore
from traits.api import Dict # type: ignore
import matplotlib.pyplot as plt


MAXNSOURCES = 9

# Raumdimensionen
YMIN = -1.5
YMAX = 1.5
XMIN = -1.5
XMAX = 1.5
Z = 2.0
INCREMENT = 3/63


class ConfigUMA(DatasetSyntheticConfig):
    """Configuration for the UMA-16 microphone array.

    Based on an example from the acoupipe documentation.
    https://adku1173.github.io/acoupipe/contents/jupyter/modify.html

    UMA aperture: 0.178 m
    """

    fft_params = Dict({
                    'block_size' : 256,
                    'overlap' : '50%',
                    'window' : 'Hanning',
                    'precision' : 'complex64'},
                desc='FFT parameters')

    def create_mics(self):
        uma_file = Path(ac.__file__).parent / 'xml' / 'minidsp_uma-16.xml'
        return ac.MicGeom(from_file=uma_file)

    def create_grid(self):
        return ac.RectGrid(y_min=YMIN, y_max=YMAX, x_min=XMIN, x_max=XMAX, z=Z, increment=INCREMENT)

    def create_location_sampler(self):
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



# config = ConfigUMA() 

# dataset_uma = DatasetSynthetic(config=config)

# plt.figure()
# plt.scatter(
#     dataset_uma.config.mics.mpos[0],
#     dataset_uma.config.mics.mpos[1])
# plt.show()
