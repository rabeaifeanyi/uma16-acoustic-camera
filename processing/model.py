import numpy as np
from acoupipe.datasets.synthetic import DatasetSynthetic # type: ignore
import time


class ModelProcessorOLD:
    def __init__(self, frame_width, frame_height, config):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.config = config
    
    def get_uma16_dummy_data(self):
        """
        Dummy data creation for testing.
        """
        noise_level = 0.05
        dataset_uma = DatasetSynthetic(config=self.config)
        data_generator = dataset_uma.generate(f=5000, num=3, size=100, split='validation', features = ['loc'] )

        while True:
            data = next(data_generator)
            source_locations = data['loc'][:2].T
            strengths = np.random.rand(source_locations.shape[0]) * 20 + 60  # Random dB values from 60 to 80

            noisy_loc = source_locations + np.random.normal(scale=noise_level, size=source_locations.shape)
            
            time.sleep(1)
            print(noisy_loc)
            
            return {
                'x': noisy_loc[:, 0].tolist(),
                'y': noisy_loc[:, 1].tolist(),
                's': strengths.tolist()
            }

    
    def get_uma16_data(self):
        pass


class ModelProcessor:
    def __init__(self, frame_width, frame_height, config):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.config = config
        self.noise_level = 0.05
        self.dataset_uma = DatasetSynthetic(config=self.config)
        self.data_generator = self.dataset_uma.generate(f=5000, num=3, size=100, split='validation', features = ['loc'], progress_bar=False)
    
    def get_uma16_dummy_data(self):

        data = next(self.data_generator)
        source_locations = data['loc'][:2].T
        strengths = np.random.rand(source_locations.shape[0]) * 20 + 60  # Random dB values from 60 to 80

        noisy_loc = source_locations + np.random.normal(scale=self.noise_level, size=source_locations.shape)
        
        return {
            'x': noisy_loc[:, 0].tolist(),
            'y': noisy_loc[:, 1].tolist(),
            's': strengths.tolist()
        }

    
    def get_uma16_data(self):
        pass

