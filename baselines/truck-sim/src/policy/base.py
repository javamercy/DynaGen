from src.simulator.location import Depot, Customer
from src.simulator.request import Request
from src.simulator.truck import Truck


class BasePolicy(object):
    def __init__(self,
                 depots: list[Depot],
                 customers: list[Customer],
                 trucks: list[Truck],
                 requests: list[Request],
                 end_time: float):
        self.depots: list[Depot] = depots
        self.customers: list[Customer] = customers
        self.trucks: list[Truck] = trucks
        self.requests: list[Request] = requests
        self.end_time: float = end_time

    def init(self):
        raise NotImplementedError

    def update(self, time: float):
        raise NotImplementedError
