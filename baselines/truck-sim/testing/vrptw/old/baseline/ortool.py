from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def get_solution(num_vehicles, manager, routing, solution) -> int:
    time_dim = routing.GetDimensionOrDie("Time")

    # Prints solution on console.
    # print(f"Objective: {solution.ObjectiveValue()}")
    total_distance = 0
    max_time = 0

    results: list[tuple[int, float, float]] = []

    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0

        while not routing.IsEnd(index):
            time_var = time_dim.CumulVar(index)
            plan_output += (f" {manager.IndexToNode(index)} "
                            f" Time({solution.Min(time_var)},{solution.Max(time_var)})"
                            f" -> ")
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)

        time_var = time_dim.CumulVar(index)

        plan_output += (f"{manager.IndexToNode(index)}"
                        f" Time({solution.Min(time_var)},{solution.Max(time_var)})\n")
        plan_output += f"Route Distance: {route_distance}, Duration: {solution.Value(time_var)}\n"

        print(plan_output)

        max_time = max(solution.Value(time_var), max_time)
        total_distance += route_distance

    print(f"Time to finish all routes: {max_time}")
    print(f"Total Distance of all routes: {total_distance}")
    return total_distance


def vrptw(Customers: list[int],
          Demand: list[int],
          LowerTimeWindow: list[int],
          UpperTimeWindow: list[int],
          Distance: list[list[int]],
          ServiceTime: list[int],
          Capacity: int,
          num_vehicles: int,
          M: int):
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(Customers), num_vehicles, 0)

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # Define distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return Distance[from_node][to_node]

    # Register distance callback
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return Demand[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [Capacity] * num_vehicles,  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Add Time Windows constraint
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return ServiceTime[from_node] + Distance[from_node][to_node]

    time_callback_index = routing.RegisterTransitCallback(time_callback)

    routing.AddDimension(
        time_callback_index,
        M,  # allow waiting time
        M,  # maximum time per vehicle
        True,  # don't force start cumul to zero
        'Time')

    time_dimension = routing.GetDimensionOrDie('Time')
    time_dimension.SetGlobalSpanCostCoefficient(100_000)

    # Add time window constraints for each location except depot
    for location_idx, customer in enumerate(Customers):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(
            LowerTimeWindow[location_idx],
            UpperTimeWindow[location_idx])

    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    # search_parameters.time_limit.FromSeconds(60)
    search_parameters.time_limit.FromSeconds(1)

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Return the minimum total distance if a solution is found
    if solution:
        return get_solution(num_vehicles, manager, routing, solution)
    else:
        print("no solution found")
        return M  # Return large value if no solution found


# Example usage:
if __name__ == "__main__":
    # Example data
    Customers = [0, 1, 2, 3]  # 0 is depot
    Demand = [0, 10, 20, 15]  # demand for each customer
    LowerTimeWindow = [0, 50, 30, 40]
    UpperTimeWindow = [200, 120, 90, 110]
    Distance = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    ServiceTime = [0, 10, 15, 12]
    Capacity = 100
    num_vehicles = 2
    M = 1000

    result = vrptw(
        Customers,
        Demand,
        LowerTimeWindow,
        UpperTimeWindow,
        Distance,
        ServiceTime,
        Capacity,
        num_vehicles,
        M
    )
    print(f"Minimum total distance: {result}")