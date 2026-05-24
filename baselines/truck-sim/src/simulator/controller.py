import random
import logging

from src.policy.base import BasePolicy

from src.simulator.common import Pos
from src.simulator.location import Location, Depot, Customer
from src.simulator.request import Request
from src.simulator.trailer import Trailer
from src.simulator.truck import Truck

from src.visualization.viz import Viz


class Controller(object):
    """
    Controller class for the simulation
    Attributes:
        depots: a list of Depots
        trucks: a list of Trucks
        customers: a list of Customers
        requests: a list of Orders
        end_time: the end of the day for the simulation
        _time: the current time of the simulation
    """

    def __init__(self,
                 dataset: dict,
                 policy,
                 animated: bool):
        """
        Constructor for the Controller class
        Args:
            dataset: the dataset used to initialize the simulation
            policy: the policy used to make decisions
            animated: either use visualization or don't
        """

        self.depots: list[Depot] = \
            [Depot(Pos(depot[0], depot[1])) for depot in dataset["depots"]]

        self.customers: list[Customer] = \
            [Customer(Pos(customer[0], customer[1])) for customer in dataset["customers"]]

        self.trucks: list[Truck] = \
            [Truck(self.depots[truck["start_depot"]], dataset["truck speed"]) for truck in dataset["trucks"]]

        self.requests = \
            [Request(
                self.depots[request["src"]],
                self.customers[request["dst"]],
                available_time=request["available_time"],
                loading_duration=request["loading_duration"],
                unloading_duration=request["unloading_duration"]) for request in dataset["requests"]]

        self.end_time: float = dataset["end of day"]

        self._policy: BasePolicy = policy(
            self.depots,
            self.customers,
            self.trucks,
            self.requests,
            self.end_time)

        self._viz: Viz | None
        if animated:
            self._viz = Viz(
                self.depots,
                self.customers,
                self.trucks,
                self.requests,
                self.end_time,
                dataset['service area'])

        self._time: float = 0.0
        self._animated = animated

    @classmethod
    def full_reset(cls):
        # kind of hacky
        # not sure if there is a better way to make it easier to share all depots with trucks
        Location.reset()
        Request.reset()
        Trailer.reset()
        Truck.reset()

    def get_fig(self):
        assert self._viz is not None
        return self._viz.fig

    def update(self, time: float):
        """
        Move trucks to the given time, then, assign requests to trucks.
        Trucks need to wait until the end of the current timestep to get new requests.
        """
        self._time = time

        for truck in self.trucks:
            truck.update(self._time)

        logging.debug(f"Timestep: {self._time}/{self.end_time}\n"
                      f"- Requests left: {len(self.requests)}\n")

        self._policy.update(self._time)

        if self._animated:
            self._viz.update(self._time)
