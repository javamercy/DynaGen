import random
import numpy as np

from ..ortool import dvrp
from ..interface import truck_num_scaling


def truncated_gaussian(mean: float, sigma: float) -> float:
    return min(mean + sigma*2, max(mean - sigma*2, random.gauss(mean, sigma)))


class GetData:
    def __init__(self, size):
        self.size = size

        self.n_late = round(size * 0.3)
        self.n_early = size - self.n_late
        assert self.n_early > 0

    def generate_instance(self):
        coordinates = np.random.rand(self.size, 2)

        early_times = [-1.0 for _ in range(self.n_early)]
        late_times = sorted([truncated_gaussian(1.5, 0.5) for _ in range(self.n_late)])
        arrival_times = np.array(early_times + late_times)

        return (coordinates,
                arrival_times,
                dvrp(truck_num_scaling(self.size), coordinates, arrival_times))
