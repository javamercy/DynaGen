import random
import numpy as np

from ..ortool import dvrptw
from ..interface import truck_num_scaling_1
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

        cur_offset = 0
        time_windows = [(0, 1_000_000)]

        for i in range(1, self.size):
            if np.random.random() < 0.5:
                time_windows.append((max(0, arrival_times[i]), 1_000_000))
                continue

            earliest_arrive = np.linalg.norm(coordinates[i,:] - coordinates[i,:])
            earliest_arrive += max(0, arrival_times[i])

            service_time = truncated_gaussian(2.5, 1.0)  # time the request will be available for

            arrival = earliest_arrive + cur_offset + np.random.uniform(0, 2)
            cur_offset += 0.02

            time_windows.append((arrival, arrival + service_time))

        return (coordinates,
                arrival_times,
                time_windows,
                dvrptw(truck_num_scaling_1(self.size), coordinates, arrival_times, time_windows),
                dvrptw(truck_num_scaling(self.size), coordinates, arrival_times, time_windows))
