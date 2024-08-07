import numpy as np
from config_uma16 import ConfigUMA
from acoupipe.datasets.synthetic import DatasetSynthetic
import time


class ModelProcessor:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height

    def get_data(self):
        num = np.random.randint(1, 9)
        x_positions = np.random.randint(1, self.frame_width, num)
        y_positions = np.random.randint(1, self.frame_height, num)
        strength = np.random.rand(num)*100
        return {'x': x_positions, 'y': y_positions, 's':strength}
    
    def create_uma16_data(self):
        """
        Dummy data creation for testing.
        """
        noise_level = 0.05
        config = ConfigUMA() 
        dataset_uma = DatasetSynthetic(config=config)
        data_generator = dataset_uma.generate(f=5000, num=3, size=100, split='validation', features = ['loc'] )

        data = next(data_generator)
        source_locations = data['loc'][:2].T
        strengths = np.random.rand(source_locations.shape[0]) * 20 + 60  # Random dB values from 60 to 80

        while True:
            if np.random.choice([0,1]):
                data = next(data_generator)
                source_locations = data['loc'][:2].T
                strengths = np.random.rand(source_locations.shape[0]) * 20 + 60
            
            noisy_loc = source_locations + np.random.normal(scale=noise_level, size=source_locations.shape)
            noisy_loc = [{'x': x, 'y': y, 'strength': s} for (x, y), s in zip(noisy_loc, strengths)]
            yield noisy_loc


