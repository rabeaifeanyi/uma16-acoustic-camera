import numpy as np
import acoular as ac # type: ignore
import matplotlib.pyplot as plt
from acoupipe.datasets.synthetic import DatasetSynthetic # type: ignore

class ModelProcessor:
    """Model processor for the UMA-16 microphone array.
    """
    def __init__(self, frame_width, frame_height, config, mic_index):
        """Initialize the model processor.
        
        Args:
            frame_width (int): Width of the frame.
            frame_height (int): Height of the frame.
            config (ConfigUMA): Configuration for the UMA-16 microphone array.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.mic_index = mic_index
        
        self.dev = ac.SoundDeviceSamplesGenerator(device=self.mic_index, numchannels=16)
        self.recording_time = 1
        self.dev.numsamples = int(self.recording_time * self.dev.sample_freq)
        self.t = np.arange(self.dev.numsamples) / self.dev.sample_freq
        
        # Dummy data
        self.config = config
        self.noise_level = 0.05
        self.dataset_uma = DatasetSynthetic(config=self.config)
        self.data_generator = self.dataset_uma.generate(f=5000, 
                                                        num=3, 
                                                        size=100, 
                                                        split='validation', 
                                                        features = ['loc'], 
                                                        progress_bar=False)
    
    def get_uma_data(self):
        signal = ac.tools.return_result(self.dev, num=256)
        return {
            'x': self.t.tolist(), 
            'y': signal[:,0].tolist()
        }

    def get_uma16_dummy_data(self):
        """Get dummy data for the UMA-16 microphone array.
        """
        try:
            data = next(self.data_generator)
        except StopIteration:
            self.data_generator = self.dataset_uma.generate(f=5000, 
                                                            num=3, 
                                                            size=100, 
                                                            split='validation', 
                                                            features=['loc'], 
                                                            progress_bar=False)
            data = next(self.data_generator)

        source_locations = data['loc'][:2].T
        strengths = np.random.rand(source_locations.shape[0]) * 20 + 60  # Random values from 60 to 80

        noisy_loc = source_locations + np.random.normal(scale=self.noise_level, size=source_locations.shape)

        return {
            'x': noisy_loc[:, 0].tolist(),
            'y': noisy_loc[:, 1].tolist(),
            's': strengths.tolist()
        }


        
