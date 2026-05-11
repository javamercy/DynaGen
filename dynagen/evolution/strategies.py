from enum import StrEnum


class Strategy(StrEnum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"


_PARENT_COUNTS: dict[Strategy, int] = {
    Strategy.S1: 1,
    Strategy.S2: 1,
    Strategy.S3: 3,
}


def parent_count(strategy: Strategy) -> int:
    return _PARENT_COUNTS[strategy]
