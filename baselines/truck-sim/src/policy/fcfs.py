from src.policy.base import BasePolicy

from src.simulator.common import TruckStatus, RequestStatus
from src.simulator.location import Depot, Customer
from src.simulator.request import Request
from src.simulator.truck import Truck


class FirstComeFirstServed(BasePolicy):
    """
    Policy of first come, first served
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
        self.init()

    def init(self):
        requests = self.requests.copy()
        for request in requests:
            if request.is_available(0.0):
                self.cur_requests.append(request)
                self.requests.remove(request)

        # zipping lists of different len with stop at the shorter list
        for truck, request in zip(self.trucks, self.cur_requests):
            truck.requests.append(request)
            request.accept()

        for _ in range(min(len(self.trucks), len(self.cur_requests))):
            self.cur_requests.pop(0)

    def update(self, time: float):
        requests = self.requests.copy()
        for request in requests:
            if request.is_available(time):
                self.cur_requests.append(request)
                self.requests.remove(request)

        available_trucks = []
        for truck in self.trucks:
            if truck.status == TruckStatus.IDLE:
                available_trucks.append(truck)

        for truck, request in zip(available_trucks, self.cur_requests):
            assert request.status == RequestStatus.AVAILABLE
            truck.requests.append(request)
            request.accept()

        for _ in range(min(len(available_trucks), len(self.cur_requests))):
            self.cur_requests.pop(0)
