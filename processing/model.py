import numpy as np


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
