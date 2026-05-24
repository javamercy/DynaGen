import numpy as np

from ..interface import truck_num_scaling
from ..ortool import vrp


class GetData:
    def __init__(self, size):
        self.size = size

    def generate_instance(self):
        coordinates = np.random.rand(self.size, 2)
        distances = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)
        return (coordinates,
                distances,
                vrp(truck_num_scaling(self.size), coordinates))
