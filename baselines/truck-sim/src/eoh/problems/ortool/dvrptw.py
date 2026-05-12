from .utils import *


def dvrptw(num_vehicles, locations, arrival_times, time_windows):
    manager, routing = ortool_init(num_vehicles, locations)

    dimension = routing.GetDimensionOrDie("Distance")
    dimension.SetGlobalSpanCostCoefficient(M)
    
    penalty = 1_000_000 * M
    for node in range(1, len(locations)):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Add time window constraints based on arrival_time
    for location_idx, (arrival, time_window) in enumerate(zip(arrival_times, time_windows)):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)

        lower = round(1000 * max(0, max(arrival, time_window[0])))
        upper = round(1000 * time_window[1])

        dimension.CumulVar(index).SetRange(lower, upper)

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
