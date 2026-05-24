import itertools

from src.simulator.common import Pos


class Trailer(object):
    """
    Trailer class stores information about each trailer
    """
    id_iter = itertools.count()
    instances = []

    def __init__(self, pos):
        assert pos is not None

        self.id: int = next(Trailer.id_iter)
        self.pos: Pos = pos

        self.instances.append(self)

    @classmethod
    def reset(cls):
        cls.id_iter = itertools.count()
        cls.instances.clear()
