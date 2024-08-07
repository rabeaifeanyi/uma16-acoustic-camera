import acoular as ac
from acoupipe.datasets.synthetic import DatasetSyntheticConfig
from pathlib import Path
from scipy.stats import uniform, norm
import acoupipe.sampler as sp
from traits.api import Bool, Dict, Float, Instance

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
        uma_file = Path(ac.__file__).parent / 'xml' / 'minidsp_uma16.xml'
        return ac.MicGeom(from_file=uma_file)

    def create_grid(self):
        return ac.RectGrid(y_min=-1.5, y_max=1.5, x_min=-1.5, x_max=1.5,
                                    z=2.0, increment=3/63)

    def create_location_sampler(self):
        location_sampler = sp.LocationSampler(
            random_var = (uniform(-1.5,3),uniform(-1.5,3),uniform(2.0,0)),
            nsources = self.max_nsources,
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
