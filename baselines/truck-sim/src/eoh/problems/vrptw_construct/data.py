import random
import numpy as np

from ..interface import truck_num_scaling_1
from ..interface import truck_num_scaling
from ..ortool import vrptw


def truncated_gaussian(mean: float, sigma: float) -> float:
    return min(mean + sigma*2, max(mean - sigma*2, random.gauss(mean, sigma)))

# give 80% of orders a time window
# 50% percent of orders having time window after

class GetData:
    def __init__(self, size):
        self.size = size

    def generate_instance(self):
        coordinates = np.random.rand(self.size, 2)

        cur_offset = 0

        time_windows = [(0, 1_000_000)]
        for i in range(1, self.size):
            if np.random.random() < 0.5:
                time_windows.append((0, 1_000_000))
                continue

            earliest_arrive: float = np.linalg.norm(coordinates[i,:] - coordinates[i,:])

            service_time: float = truncated_gaussian(2.5, 1.0)  # time the request will be available for

            arrival = earliest_arrive + cur_offset + np.random.uniform(0, 2)
            cur_offset += 0.02

            time_windows.append((arrival, arrival + service_time))

        return (coordinates,
                time_windows,
                vrptw(truck_num_scaling_1(self.size), coordinates, time_windows),
                vrptw(truck_num_scaling(self.size), coordinates, time_windows))

