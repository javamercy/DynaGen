import itertools

from src.simulator.common import Pos


class Location(object):
    """
    Location is a class for customer and depot locations
    Attributes:
        id: unique sequential id
        pos: the x and y position of the location
        _trucks: the list of trucks currently at this location
        _trailers: the list of trailers currently at this location
    """
    id_iter = itertools.count()

    def __init__(self, pos: Pos):
        assert pos is not None

        self.id: int = next(Location.id_iter)
        self.pos: Pos = pos
        self._trucks = []
        self._trailers = []

    @classmethod
    def reset(cls):
        cls.id_iter = itertools.count()

    def dock_truck(self, truck):
        self._trucks.append(truck)

    def dock_trailer(self, trailer):
        pass

    def undock_truck(self, truck):
        self._trucks.remove(truck)

    def undock_trailer(self, trailer):
        pass


class Depot(Location):
    instances = []

    def __init__(self, pos: Pos):
        super().__init__(pos)

        self.instances.append(self)

    @classmethod
    def reset(cls):
        super().reset()
        cls.instances.clear()


class Customer(Location):
    instances = []

    def __init__(self, pos: Pos):
        super().__init__(pos)

        self.instances.append(self)

    @classmethod
    def reset(cls):
        super().reset()
        cls.instances.clear()
