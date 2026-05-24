import logging
import itertools

from collections import deque

from src.simulator.common import Pos, TruckStatus
from src.simulator.location import Location, Depot
from src.simulator.request import Request


class Truck(object):
    """
    Truck is a class to record information for each truck
    Attributes:
        id: unique sequential id
        pos: the current position of the truck
        status: idle, moving, unloading
        location: the current docked location of the truck (truck is always docked at the start)

        _dock_time: the time when the truck last docked

        _last_time: the truck _time at current state
        _time_step: the difference from _last_time to the new time
        _time_left: the amount of time left until we arrive at the new time
        _dist_tr: total distance traveled (meters)
        _time_idle: time spent idle (minutes)
        _time_moving: time spent moving (minutes)
        _time_waiting: time spent unloading (minutes)
        _deliveries_made: number of deliveries made

        request: the current request
        requests: a list of Requests
        requests_done:
    """
    id_iter = itertools.count()
    instances = []

    def __init__(self, start: Location, speed: float):
        assert start is not None

        self.id: int = next(Truck.id_iter)
        self.pos: Pos = Pos(start.pos.x, start.pos.y)
        self.status: TruckStatus = TruckStatus.IDLE
        self.location: Location | None = start
        self.location.dock_truck(self)

        self._dock_time: float = 0.0

        self._truck_speed: float = speed
        self._last_time: float = 0.0
        self._time_step: float = 0.0
        self._time_left: float = 0.0
        self._dist_tr: float = 0.0
        self._time_idle: float = 0.0
        self._time_moving: float = 0.0
        self._time_waiting: float = 0.0
        self._deliveries_made: int = 0

        self.request: Request | None = None
        self.requests: deque[Request] = deque()
        self.requests_done = []

        self.instances.append(self)

    @classmethod
    def reset(cls):
        cls.id_iter = itertools.count()
        cls.instances.clear()

    def docked(self) -> bool:
        return self.location is not None

    def _dock(self, location: Location):
        assert self.location is None
        self.location = location
        self.location.dock_truck(self)

    def _undock(self):
        assert self.location is not None
        self.location.undock_truck(self)
        self.location = None

    def _used_time(self):
        return self._time_step - self._time_left

    def _cur_time(self):
        return self._last_time + self._used_time()

    def update(self, time: float):
        assert time >= self._last_time

        self._time_left = time - self._last_time
        self._time_step = time - self._last_time
        while self._time_left > 0:
            match self.status:
                case TruckStatus.IDLE:
                    # if TruckStatus.MOVING then check if we have any requests, otherwise just idle
                    assert self.request is None
                    if len(self.requests) == 0:
                        self._update_idle()
                    else:
                        self._start_request()

                case TruckStatus.MOVING:
                    # if TruckStatus.MOVING then move until
                    # - we run out of _time
                    # - we arrive at the next position, then we switch to waiting
                    assert self.request is not None
                    self._update_moving()
                    if self._time_left > 0.0:
                        self._done_moving()

                case TruckStatus.WAITING:
                    # if TruckStatus.WAITING then wait until
                    # - we run out of _time
                    # - the cargo is done being loaded/unloaded, then we check if the request is done
                    assert self.request is not None
                    self._update_wait()
                    if self._time_left > 0.0:
                        self._done_waiting()

        self._last_time = time

    def _set_pos(self, x: float, y: float):
        # TODO: if trailer attached then also update the location of the trailer
        self.pos.x = x
        self.pos.y = y

    def _update_idle(self):
        if not self.docked():
            assert len(Depot.instances) > 0
            nearest_depot: Depot = Depot.instances[0]
            nearest_depot_dist = self.pos.dist(nearest_depot.pos)
            for depot in Depot.instances:
                if self.pos.dist(depot.pos) < nearest_depot_dist:
                    nearest_depot = depot
                    nearest_depot_dist = self.pos.dist(depot.pos)

            want_dist = nearest_depot_dist
            can_dist = self._time_left * self._truck_speed

            # TODO: find a better way to handle this
            self._dist_tr += min(want_dist, can_dist)
            self._time_moving += min(want_dist, can_dist) / self._truck_speed

            if want_dist >= can_dist:
                self._set_pos(self.pos.x + (nearest_depot.pos.x - self.pos.x) * (can_dist / want_dist),
                              self.pos.y + (nearest_depot.pos.y - self.pos.y) * (can_dist / want_dist))
            else:
                self._set_pos(nearest_depot.pos.x,
                              nearest_depot.pos.y)
                self._time_left -= want_dist / self._truck_speed

                self._dock(nearest_depot)
                self._dock_time = self._last_time + self._used_time()

        self._time_idle += self._time_left
        self._time_left = 0.0

    def _start_request(self):
        if self.docked():
            self._undock()

        self.status = TruckStatus.MOVING
        self.request = self.requests.popleft()
        self.request.start(self._cur_time())

    def _update_moving(self):
        pos: Pos = self.request.next_pos()

        want_dist = self.pos.dist(pos)
        can_dist = self._time_left * self._truck_speed

        self._dist_tr += min(want_dist, can_dist)
        self._time_moving += min(want_dist, can_dist) / self._truck_speed

        if want_dist >= can_dist:
            self._set_pos(self.pos.x + (pos.x - self.pos.x) * (can_dist / want_dist),
                          self.pos.y + (pos.y - self.pos.y) * (can_dist / want_dist))
            self._time_left = 0.0

        else:
            self._set_pos(pos.x,
                          pos.y)
            self._time_left -= want_dist / self._truck_speed

    def _done_moving(self):
        self._dock(self.request.next_location())
        self.request.arrive_at_location()
        self.status = TruckStatus.WAITING

    def _update_wait(self):
        self._time_left = self.request.wait_time(self._time_left)
        self._time_waiting += self._used_time()

    def _done_waiting(self):
        self._undock()
        self.request.depart_or_done(self._cur_time())

        if not self.request.is_complete():
            self.status = TruckStatus.MOVING

        else:
            self._deliveries_made += 1
            self.status = TruckStatus.IDLE
            self.requests_done.append(self.request)
            self.request = None

    def print_stats(self):
        result = (f"Truck: {self.id}"
                  f"\n- Status: {self.status}"
                  f"\n- Idle: {self._time_idle}"
                  f"\n- Moving: {self._time_moving}"
                  f"\n- Waiting: {self._time_waiting}")

        if self.request is None and len(self.requests) == 0 and len(self.requests_done) > 0 and self.location is not None:
            # result += f"\n- Done at: {self.requests_done[-1].complete_time}"
            result += f"\n- Done at: {self._dock_time}"

        if self.request is not None:
            result += (f"\n- Time: {self._last_time}"
                       f"\n- Time Left: {self.request.duration_left(self.pos, self._truck_speed)}")

        result += "\n"
        logging.info(result)

    def get_stats(self):
        return {
            "count": self._deliveries_made,
            "time": self._dock_time,
            "distance": self._dist_tr,
        }

    def _future_request_durations(self, requests: list[Request] | None = None) -> list[float]:
        """
        Returns:
            A list of durations for requests that the truck has queued into the future
        """
        if requests is None:
            requests = list(self.requests)

        # get the first request and find the amount of _time to finish it
        if self.request is not None:
            cur: Request = self.request

        else:
            if len(self.requests) == 0:
                return []
            cur: Request = requests.pop(0)

        # find the amount of time to finish the rest of the requests
        durations: list[float] = [cur.duration_left(self.pos, self._truck_speed)]
        for nxt in requests:
            durations.append(nxt.duration_left(cur.destination.pos, self._truck_speed))
            cur = nxt

        # the time to get back to the depot
        durations.append(min(cur.destination.pos.dist(depot.pos) for depot in Depot.instances) / self._truck_speed)

        return durations

    def time_to_finish(self, requests: list[Request] | None = None) -> float:
        """
        Calculate the time when truck will finish its requests
        Returns:
            - the time when truck will finish all its requests
        """
        return sum(self._future_request_durations(requests))

    def get_request_schedule(self) -> list[tuple[int, float, float]]:
        result: list[tuple[int, float, float]] = []

        # get the schedule for completed tasks
        for request in self.requests_done:
            assert request.start_time != -1
            assert request.complete_time != -1

            result.append((request.id,
                           request.start_time,
                           request.complete_time - request.start_time))

        last_time = self._last_time
        future_request_durations = self._future_request_durations()

        # go back to capture the full duration of the current task
        if self.request is not None:
            assert self.request.start_time != -1
            assert len(future_request_durations) > 0
            future_request_durations[0] += last_time - self.request.start_time

            result.append((self.request.id,
                           self.request.start_time,
                           future_request_durations[0]))
            last_time = self.request.start_time + future_request_durations[0]

            future_request_durations.pop(0)

        # get the durations of future tasks
        for request, request_duration in zip(self.requests, future_request_durations):
            result.append((request.id,
                           last_time,
                           request_duration))
            last_time += request_duration

        return result
