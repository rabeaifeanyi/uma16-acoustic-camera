import numpy as np
from acoupipe.datasets.synthetic import DatasetSynthetic # type: ignore

class ModelProcessor:
    """Model processor for the UMA-16 microphone array.
    """
    def __init__(self, frame_width, frame_height, config):
        """Initialize the model processor.
        
        Args:
            frame_width (int): Width of the frame.
            frame_height (int): Height of the frame.
            config (ConfigUMA): Configuration for the UMA-16 microphone array.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.config = config
        self.noise_level = 0.05
        self.dataset_uma = DatasetSynthetic(config=self.config)
        self.data_generator = self.dataset_uma.generate(f=5000, num=3, size=100, split='validation', features = ['loc'], progress_bar=False)
    
    def get_uma16_dummy_data(self):
        """Get dummy data for the UMA-16 microphone array.
        """
        try:
            data = next(self.data_generator)
        except StopIteration:
            # Reinitialize the data generator if it is exhausted
            self.data_generator = self.dataset_uma.generate(f=5000, num=3, size=100, split='validation', features=['loc'], progress_bar=False)
            data = next(self.data_generator)

        source_locations = data['loc'][:2].T
        strengths = np.random.rand(source_locations.shape[0]) * 20 + 60  # Random dB values from 60 to 80

        noisy_loc = source_locations + np.random.normal(scale=self.noise_level, size=source_locations.shape)

        return {
            'x': noisy_loc[:, 0].tolist(),
            'y': noisy_loc[:, 1].tolist(),
            's': strengths.tolist()
        }

    def get_model_data(self):
        pass

