import numpy as np
from ..ortool import tsp


class GetData:
    def __init__(self, size):
        self.size = size

    def generate_instance(self):
        coordinates = np.random.rand(self.size, 2)
        distances = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)

        return (coordinates,
                distances,
                tsp(coordinates))
