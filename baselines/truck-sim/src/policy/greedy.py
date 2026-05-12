import math

from src.policy.base import BasePolicy
from src.simulator.common import RequestStatus
from src.simulator.location import Depot, Customer
from src.simulator.request import Request
from src.simulator.truck import Truck


def argmin(a):
    return min(range(len(a)), key=lambda x: a[x])


def argmax(a):
    return max(range(len(a)), key=lambda x: a[x])


class CheapestInsertion(BasePolicy):
    """
    Policy of cheapest insertion
    Attributes:
        depots: a list of Depot
        customers: a list of Customer
        trucks: a list of Truck
        requests: a list of Request
    """
    def __init__(self,
                 depots: list[Depot],
                 customers: list[Customer],
                 trucks: list[Truck],
                 requests: list[Request],
                 end_time: float):

        super().__init__(
            depots,
            customers,
            trucks,
            requests,
            end_time)

        self.cur_requests: list[Request] = []
        self.time: float = -1.0
        self.init()

    def init(self):
        requests = self.requests.copy()
        for request in requests:
            if request.is_available(0.0):
                self.cur_requests.append(request)
                self.requests.remove(request)

        # assign each truck the shortest request they can take
        # TODO: better way to initially schedule the requests
        for truck in self.trucks:
            best_req = self.cur_requests[0]
            best_req_score = truck.time_to_finish([best_req])

            for request in self.cur_requests:
                if truck.time_to_finish([request]) < best_req_score:
                    best_req = request
                    best_req_score = truck.time_to_finish([request])

            self.cur_requests.remove(best_req)
            truck.requests.append(best_req)
            best_req.accept()

    def _assign_req(self, request: Request):
        shortest_durations: list[float] = []
        shortest_durations_loc: list[int] = []

        # get the best possible place to insert the new request in each truck's request queue
        for truck in self.trucks:
            shortest_duration = math.inf
            shortest_duration_loc = -1
            requests = list(truck.requests)  # also creates a shallow copy

            for loc in range(len(truck.requests) + 1):
                test_requests = requests.copy()
                test_requests.insert(loc, request)

                duration = truck.time_to_finish(test_requests)
                if duration < shortest_duration:
                    shortest_duration = duration
                    shortest_duration_loc = loc

            shortest_durations.append(shortest_duration)
            shortest_durations_loc.append(shortest_duration_loc)

        original_durations: list[float] = [truck.time_to_finish() for truck in self.trucks]
        workday_end: float = max(original_durations)

        if self.end_time - self.time < min(shortest_durations):
            # unable to accept request (minimum end time is still after the work day is done)
            request.reject()

        elif workday_end < min(shortest_durations):
            # minimize work day (minimize the _time that the latest truck finishes)
            idx = argmin(shortest_durations)
            self.trucks[idx].requests.insert(shortest_durations_loc[idx], request)
            request.accept()

        else:
            # minimize added _time if there are multiple that don't increase the work day
            idx_lst = []
            for i, duration in enumerate(shortest_durations):
                if duration <= workday_end:
                    idx_lst.append(i)

            min_duration_inc: float = math.inf
            min_duration_inc_idx: int = -1
            for idx in idx_lst:
                duration_inc = shortest_durations[idx] - original_durations[idx]
                if duration_inc < min_duration_inc:
                    min_duration_inc = duration_inc
                    min_duration_inc_idx = idx

            self.trucks[min_duration_inc_idx].requests.insert(shortest_durations_loc[min_duration_inc_idx], request)
            request.accept()

    def update(self, time: float):
        """
        Cheapest insertion will attempt to insert the new request
        - make sure that the early requests are done
        - minimize the _time when the last truck finishes delivery
        - minimize total _time for all the trucks (tiebreaker)
        - to distribute early requests: just perform the cheapest insertion until done (does not scale well)
        """
        requests = self.requests.copy()
        for request in requests:
            if request.is_available(time):
                self.cur_requests.append(request)
                self.requests.remove(request)

        self.time = time
        for request in self.cur_requests:
            assert request.status == RequestStatus.AVAILABLE
            self._assign_req(request)

        self.cur_requests.clear()
