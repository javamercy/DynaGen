import numpy as np

from scipy.spatial import distance_matrix

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


M = 1_000_000


def get_solution(num_vehicles, manager, routing, solution) -> dict:
    distances = []
    routes = []
    missed = 0
    
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if solution.Value(routing.NextVar(node)) == node:
            missed += 1

    for vehicle_id in range(num_vehicles):
        route = []
        index = routing.Start(vehicle_id)
        route_distance = 0

        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))

            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)

        route.append(manager.IndexToNode(index))

        distances.append(route_distance / 1000)
        routes.append(route)

    # if missed != 0:
    #     return None

    return {
        "total_distance": sum(distances),
        "max_distance": max(distances),
        "distances": distances,
        "routes": routes,
        "missed": missed,
    }


def ortool_init(num_vehicles, locations):
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(locations), num_vehicles, 0)

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    distances = distance_matrix(locations, locations) * 1000
    distances = np.array(np.rint(distances), dtype=int)

    # Define distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distances[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = "Distance"
    routing.AddDimension(
        transit_callback_index,
        M,  # slack
        M,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name,
    )

    return manager, routing


def ortool_search(routing):
    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

    # search_parameters.time_limit.FromSeconds(1)
    # search_parameters.time_limit.FromSeconds(60)
    search_parameters.time_limit.FromSeconds(5 * 60)

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    return solution



# def print_solution(data, manager, routing, solution):
#     """Prints solution on console."""
#     print(f"Objective: {solution.ObjectiveValue()}")
#     time_dimension = routing.GetDimensionOrDie("Time")
#     total_time = 0
#
#     for vehicle_id in range(data["num_vehicles"]):
#         index = routing.Start(vehicle_id)
#         plan_output = f"Route for vehicle {vehicle_id}:\n"
#
#         while not routing.IsEnd(index):
#             time_var = time_dimension.CumulVar(index)
#             plan_output += (
#                 f"{manager.IndexToNode(index)}"
#                 f" Time({solution.Min(time_var)},{solution.Max(time_var)})"
#                 " -> ")
#             index = solution.Value(routing.NextVar(index))
#
#         time_var = time_dimension.CumulVar(index)
#         plan_output += (
#             f"{manager.IndexToNode(index)}"
#             f" Time({solution.Min(time_var)},{solution.Max(time_var)})\n")
#         plan_output += f"Time of the route: {solution.Min(time_var)}min\n"
#
#         print(plan_output)
#         total_time += solution.Min(time_var)
#
#     print(f"Total time of all routes: {total_time}min")


# def get_solution(num_vehicles, manager, routing, solution) -> int:
#     time_dim = routing.GetDimensionOrDie("Time")
#
#     # Prints solution on console.
#     print(f"Objective: {solution.ObjectiveValue()}")
#     total_distance = 0
#     max_time = 0
#
#     for vehicle_id in range(num_vehicles):
#         index = routing.Start(vehicle_id)
#         plan_output = f"Route for vehicle {vehicle_id}:\n"
#         route_distance = 0
#
#         while not routing.IsEnd(index):
#             time_var = time_dim.CumulVar(index)
#             plan_output += (f" {manager.IndexToNode(index)} "
#                             f" Time({solution.Min(time_var)},{solution.Max(time_var)})"
#                             f" -> ")
#             previous_index = index
#             index = solution.Value(routing.NextVar(index))
#             route_distance += routing.GetArcCostForVehicle(
#                 previous_index, index, vehicle_id)
#
#         time_var = time_dim.CumulVar(index)
#
#         plan_output += (f"{manager.IndexToNode(index)}"
#                         f" Time({solution.Min(time_var)},{solution.Max(time_var)})\n")
#         plan_output += f"Route Distance: {route_distance}, Duration: {solution.Value(time_var)}\n"
#
#         print(plan_output)
#
#         max_time = max(solution.Value(time_var), max_time)
#         total_distance += route_distance
#
#     print(f"Time to finish all routes: {max_time}")
#     print(f"Total Distance of all routes: {total_distance}")
#     return max_time
