import math
import random
import itertools

from collections import deque

from src.simulator.location import Location, Depot, Customer
from src.simulator.common import Pos, RequestStatus


class Part(object):
    """
    A Part of the _route
    Attributes:
        location: the location we are trying to reach
        wait_time: the _time to wait when we arrive there
    """

    def __init__(self, location: Location, wait_time: float):
        self.location: Location = location
        self.wait_time: float = wait_time


class Request(object):
    """
    A class to store the contents of a delivery request
    Attributes:
        id: unique sequential id
        status: the current status of the request
        available_time: the time when the request is available (negative values indicate early request)

        source: the depot to pick up the order
        destination: the customer to drop off the order
        load_duration: time to load the order
        unload_duration: time to unload the order

        _was_early: stores if the request was early or not
        start_time: stores when the truck started the request (not when the request was queued)
        complete_time: stores when the request was fully completed

        _route: a list of Part that make up the route
    """

    id_iter = itertools.count()
    instances = []

    def __init__(self,
                 src: Depot,
                 dst: Customer,
                 available_time: float,
                 loading_duration: float,
                 unloading_duration: float):

        self.id: int = next(Request.id_iter)
        self.status: RequestStatus = RequestStatus.AVAILABLE if available_time < 0.0 else RequestStatus.UNAVAILABLE
        self.available_time = available_time

        # in the future it might be something like: truck, depot, customer and another argument for live/unlive dropoff
        self.source: Depot = src
        self.destination: Customer = dst
        self.load_duration: float = loading_duration
        self.unload_duration: float = unloading_duration

        self._was_early = available_time < 0.0
        self.start_time = -1.0
        self.complete_time = -1.0

        self._route: deque[Part] = deque([
            Part(self.source, self.load_duration),
            Part(self.destination, self.unload_duration),
        ])

        self.instances.append(self)

    @classmethod
    def reset(cls):
        cls.id_iter = itertools.count()
        cls.instances.clear()

    def accept(self):
        assert self.status == RequestStatus.AVAILABLE
        self.status = RequestStatus.ACCEPTED

    def reject(self):
        assert self.status == RequestStatus.AVAILABLE
        self.status = RequestStatus.REJECTED

    def start(self, time: float):
        self._update_started()
        self.start_time = time

    def next_pos(self) -> Pos:
        assert len(self._route) != 0
        return self._route[0].location.pos

    def next_location(self) -> Location:
        assert len(self._route) != 0
        return self._route[0].location

    def arrive_at_location(self):
        match self.status:
            case RequestStatus.STARTED:
                self._update_loading()
            case RequestStatus.MOVING:
                self._update_unloading()

    def wait_time(self, max_time: float) -> float:
        assert len(self._route) != 0

        if max_time < self._route[0].wait_time:
            self._route[0].wait_time -= max_time
            return 0.0

        else:
            max_time -= self._route[0].wait_time
            self._route[0].wait_time = 0

            return max_time

    def depart_or_done(self, time: float):
        assert len(self._route) != 0
        route = self._route.popleft()

        if len(self._route) > 0:
            self._update_moving()

        else:
            self._update_completed()
            self.complete_time = time

    def is_complete(self) -> bool:
        return self.status == RequestStatus.COMPLETED

    def is_available(self, time: float) -> bool:
        if time >= self.available_time:
            self.status = RequestStatus.AVAILABLE
            return True
        return False

    def _update_started(self):
        assert self.status == RequestStatus.ACCEPTED
        self.status = RequestStatus.STARTED

    def _update_loading(self):
        assert self.status == RequestStatus.STARTED
        self.status = RequestStatus.LOADING

    def _update_moving(self):
        assert self.status == RequestStatus.LOADING
        self._loaded = True
        self.status = RequestStatus.MOVING

    def _update_unloading(self):
        assert self.status == RequestStatus.MOVING
        self.status = RequestStatus.UNLOADING

    def _update_completed(self):
        assert self.status == RequestStatus.UNLOADING
        self.status = RequestStatus.COMPLETED

    def duration_left(self, pos: Pos, speed: float) -> float:
        time: float = 0.0
        cur: Pos = pos
        for part in self._route:
            nxt: Pos = part.location.pos
            time += part.wait_time
            time += math.sqrt((cur.x - nxt.x)**2 + (cur.y - nxt.y)**2) / speed
            cur = nxt

        return time


