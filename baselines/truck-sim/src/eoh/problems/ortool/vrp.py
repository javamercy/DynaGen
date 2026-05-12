from .utils import *


def vrp(num_vehicles, locations):
    manager, routing = ortool_init(num_vehicles, locations)

    distance_dimension = routing.GetDimensionOrDie("Distance")
    distance_dimension.SetGlobalSpanCostCoefficient(M)

    solution = ortool_search(routing)
    if solution:
        return get_solution(num_vehicles, manager, routing, solution)
