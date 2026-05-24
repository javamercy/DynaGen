from .utils import *


def tsp(locations):
    manager, routing = ortool_init(1, locations)

    distance_dimension = routing.GetDimensionOrDie("Distance")
    distance_dimension.SetGlobalSpanCostCoefficient(M)

    solution = ortool_search(routing)
    if solution:
        return get_solution(1, manager, routing, solution)
