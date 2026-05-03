from enum import StrEnum


class Strategy(StrEnum):
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    M1 = "M1"
    M2 = "M2"


_PARENT_COUNTS: dict[Strategy, int] = {
    Strategy.E1: 3,
    Strategy.E2: 3,
    Strategy.E3: 2,
    Strategy.M1: 1,
    Strategy.M2: 1,
}


def parent_count(strategy: Strategy) -> int:
    return _PARENT_COUNTS[strategy]
