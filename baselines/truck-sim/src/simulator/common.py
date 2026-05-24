from __future__ import annotations

import math

from enum import Enum


class TruckStatus(Enum):
    IDLE = 0  # truck does not a request
    MOVING = 1  # any _time the truck is moving while it has a request
    WAITING = 2  # loading/unloading


class RequestStatus(Enum):
    UNAVAILABLE = 0  # cannot be taken at the current timestep
    AVAILABLE = 1  # available to be taken
    ACCEPTED = 2  # already on some truck's queue
    STARTED = 3  # taken off the truck's queue and started
    LOADING = 4  # currently being loaded
    MOVING = 5  # currently on the move to the destination
    UNLOADING = 6  # unloading at the destination
    COMPLETED = 7  # done
    REJECTED = 8


class Pos(object):
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def dist(self, p: Pos):
        assert p is not None

        x = p.x - self.x
        y = p.y - self.y
        return math.sqrt(x ** 2 + y ** 2)
