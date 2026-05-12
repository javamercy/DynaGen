from .utils import *


def dvrp(num_vehicles, locations, arrival_times):
    manager, routing = ortool_init(num_vehicles, locations)

    dimension = routing.GetDimensionOrDie("Distance")
    dimension.SetGlobalSpanCostCoefficient(M)

    # Add time window constraints based on arrival_time
    for location_idx, arrival in enumerate(arrival_times):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        dimension.CumulVar(index).SetRange(round(max(arrival, 0) * 1000), M)

    # add time window constraints for depot
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        dimension.CumulVar(index).SetRange(0, M)

    # Instantiate route start and end times to produce feasible times.
    for i in range(num_vehicles):
        routing.AddVariableMinimizedByFinalizer(dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(dimension.CumulVar(routing.End(i)))

    solution = ortool_search(routing)
    if solution:
        return get_solution(num_vehicles, manager, routing, solution)
