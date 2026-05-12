import math
import itertools
import numpy as np

from typing import Optional
from dataclasses import dataclass
from scipy.spatial import distance_matrix
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.policy.base import BasePolicy

from src.simulator.common import Pos
from src.simulator.location import Location, Depot, Customer
from src.simulator.request import Request
from src.simulator.truck import Truck


class ORTranslation(object):
    """
    Workarounds to translate our problem into ORTools:
        - ORTools cannot reuse nodes for pickup and delivery this means
            - we need to create a new node for a depot for each request
            - when we have deliveries to the same location we need different nodes
        - Only one depot can be specified as the final destination in ORTools
            - we find the shortest distance from every customer to every depot
            - take the closer depot and remember which depot that is
    """

    def __init__(self):
        self.id_iter = itertools.count()
        self._location_to_node: dict[Location, list[int]] = {}
        self._node_to_location: list[Location] = []
        self._end_depot_mapping: list[int] = []

    def location_to_node(self, location: Location) -> int:
        if location not in self._location_to_node.keys():
            self._location_to_node[location] = []
        cur: int = next(self.id_iter)

        self._location_to_node[location].append(cur)
        self._node_to_location.append(location)
        assert self._node_to_location[cur] is location

        return cur

    def node_to_location(self, node: int) -> Location:
        return self._node_to_location[node]

    def set_end_depot_mapping(self, mapping: list[int]):
        self._end_depot_mapping = mapping

    def end_depot_mapping(self, node: int) -> Location:
        return self.node_to_location(self._end_depot_mapping[node])


@dataclass
class ORToolsData:
    """
    Convenient dataclass for instance data when using OR-Tools as solver.

    Parameters:
        ort: The translation from nodes to Locations
        # depots: The depot indexes
        distance_matrix: The distance matrix between locations.
        duration_matrix: The duration matrix between locations. This includes service times.
        num_vehicles: The number of vehicles.
        vehicle_capacities: The capacity of each vehicle.
        max_distance: The maximum distance a vehicle can travel.
        demands: The demands of each location.
        time_windows: The _time windows for each location. Optional.
        backhauls: The pickup quantity for backhaul at each location.
    """

    ort: ORTranslation
    start_depots: list[int]
    distance_matrix: list[list[int]]
    duration_matrix: list[list[int]]
    num_vehicles: int
    vehicle_capacities: list[int]
    max_distance: int
    max_time: int
    demands: list[int]
    requests: list[tuple[int, int]]
    time_windows: list[tuple[int, int]]
    backhauls: Optional[list[int]]
    truck_capacities: list[int]

    @property
    def num_locations(self) -> int:
        return len(self.distance_matrix)


def get_solution(ort: ORTranslation, data, manager, routing, solution):
    def get_str(idx: int):
        loc: Location = ort.node_to_location(manager.IndexToNode(idx))

        if isinstance(loc, Depot):
            return f"Depot({loc.id})"
        elif isinstance(loc, Customer):
            return f"Customer({loc.id})"
        else:
            return f"???({loc.id})"

    def get_last_str(idx: int):
        loc: Location = ort.end_depot_mapping(manager.IndexToNode(idx))

        if isinstance(loc, Depot):
            return f"Depot({loc.id})"
        elif isinstance(loc, Customer):
            return f"Customer({loc.id})"
        else:
            return f"???({loc.id})"

    count_dim = routing.GetDimensionOrDie("Count")
    time_dim = routing.GetDimensionOrDie("Time")

    # Prints solution on console.
    # print(f"Objective: {solution.ObjectiveValue()}")
    total_distance = 0

    results: list[tuple[int, float, float]] = []

    for vehicle_id in range(data.num_vehicles - 1):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0

        # print(f"start at: {solution.Value(time_dim.CumulVar(index)) / 60}")
        while not routing.IsEnd(index):
            time_var = time_dim.CumulVar(index)
            plan_output += (f" {manager.IndexToNode(index)}{get_str(index)} "
                            f" Time({solution.Min(time_var)},{solution.Max(time_var)})"
                            f" -> ")
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)

        count_var = count_dim.CumulVar(index)
        time_var = time_dim.CumulVar(index)

        plan_output += (f"{manager.IndexToNode(index)}{get_last_str(index)}"
                        f" Time({solution.Min(time_var)},{solution.Max(time_var)})\n")
        plan_output += f"Route Distance: {route_distance}m, Duration: {solution.Value(time_var) / 60:.5f} min\n"
        # print(plan_output)

        results.append((solution.Value(count_var), solution.Value(time_var) / 60, route_distance))
        total_distance += route_distance
    # print(f"Total Distance of all routes: {total_distance}m")

    return results


def _init_ortools_data(
        depots: list[Depot],
        customers: list[Customer],
        trucks: list[Truck],
        requests: list[Request],
        end_time: float) -> ORToolsData:

    # convert end time to seconds
    end_time: int = int(end_time * 60)
    # print(end_time)

    far_depot = Depot(Pos(100_000_000, 100_000_000))
    far_truck = Truck(far_depot, 0.0)
    depots.append(far_depot)
    trucks.append(far_truck)

    ort = ORTranslation()

    capacities: list[int] = [1 for _ in range(len(trucks))]
    max_distance: int = int(2e8)

    # service =
    # backhauls =
    # durations =

    ort_depots: list[int] = []
    ort_demands: list[int] = []  # the load taken on when visiting a node
    ort_requests: list[tuple[int, int]] = []  # the source and destination node for a delivery
    ort_locations: list[tuple[float, float]] = []
    ort_time_windows: list[tuple[int, int]] = []
    ort_loading_durations: list[tuple[int, float]] = []
    ort_unloading_durations: list[tuple[int, float]] = []

    # add dummy depot (is the closer of the two depots)
    ort.location_to_node(Depot(Pos(0.0, 0.0)))
    ort_demands.append(0)
    ort_locations.append((0.0, 0.0))
    ort_time_windows.append((0, end_time))

    depot_map: dict[Location, int] = {}
    for depot in depots:
        depot_node: int = ort.location_to_node(depot)
        depot_map[depot] = depot_node

        ort_depots.append(depot_node)
        ort_demands.append(0)
        ort_locations.append((depot.pos.x, depot.pos.y))
        ort_time_windows.append((0, end_time))

    ort_start_depots = [depot_map[truck.location] for truck in trucks]

    for request in requests:
        src: Location = request.source
        dst: Location = request.destination

        start_t = 0 if request.available_time <= 0 else math.ceil(request.available_time)

        src_node: int = ort.location_to_node(src)
        ort_demands.append(1)
        ort_locations.append((src.pos.x, src.pos.y))
        ort_time_windows.append((start_t, end_time))

        dst_node: int = ort.location_to_node(dst)
        ort_demands.append(-1)
        ort_locations.append((dst.pos.x, dst.pos.y))
        ort_time_windows.append((0, end_time))

        ort_requests.append((src_node, dst_node))
        ort_loading_durations.append((src_node, request.load_duration))
        ort_unloading_durations.append((dst_node, request.unload_duration))

    # TODO: for larger problems it doesn't actually make sense to compute the entire matrix
    #   instead we only need to know the distance from every depot to every customer

    distance_mtx = distance_matrix(ort_locations, ort_locations)
    depot_distance_mtx = np.array([distance_mtx[:, depot] for depot in ort_depots])
    ort.set_end_depot_mapping([ort_depots[idx] for idx in np.argmin(depot_distance_mtx, axis=0)])
    distance_mtx[:, 0] = np.min(depot_distance_mtx, axis=0)
    distance_mtx[0, :] = distance_mtx[:, 0]

    np.set_printoptions(suppress=True)
    duration_mtx = distance_mtx.copy() / trucks[0]._truck_speed
    # print(ort_loading_durations)
    # print(ort_unloading_durations)
    for idx, time in ort_loading_durations:
        duration_mtx[idx, :] += time
    for idx, time in ort_unloading_durations:
        duration_mtx[:, idx] += time

    # rounded to the nearest meter (ortools CANNOT handle float values for distances)
    distance_mtx = np.array(np.rint(distance_mtx), dtype=int)

    # rounded to the nearest second (ortools CANNOT handle float values for distances)
    duration_mtx *= 60
    duration_mtx = np.array(np.rint(duration_mtx), dtype=int)

    return ORToolsData(
        ort=ort,
        start_depots=ort_start_depots,
        distance_matrix=distance_mtx.tolist(),
        duration_matrix=duration_mtx.tolist(),
        num_vehicles=len(capacities),
        vehicle_capacities=capacities,
        requests=ort_requests,
        demands=ort_demands,
        time_windows=ort_time_windows,
        max_distance=max_distance,
        max_time=end_time,
        backhauls=None,
        truck_capacities=capacities)


def _solve(data: ORToolsData):
    end_depots = [0 for _ in range(data.num_vehicles)]
    end_depots[-1] = data.start_depots[-1]

    # print(data.start_depots)
    # print(end_depots)
    # print(data.ort._location_to_node)

    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(
        data.num_locations,
        data.num_vehicles,
        data.start_depots,  # [1, 1, 2],  # starting node can be either depot 1 or depot 2
        end_depots)  # [0, 0, 0])  # ending node (the depot 0 is a dummy depot to represent either depot)

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # Set arc costs equal to Euclidean distance between nodes via the distance matrix
    distance_transit_idx = routing.RegisterTransitMatrix(data.distance_matrix)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_transit_idx)

    # Set up the Distance dimension with max distance constraint
    dimension_name = "Distance"
    routing.AddDimension(
        distance_transit_idx,
        0,  # null distance slack
        data.max_distance,  # maximum distance per vehicle
        True,  # start cumul at zero
        dimension_name)

    # sets cost proportional to the "dimension span"
    # that is global_span_cost = coefficient * (Max(dimension end value) - Min(dimension start value))
    #   - dimension end value: the distances for each vehicle at the end of their trips
    #   - dimension start value: the distance that each vehicle start their trips at
    # the start value is when the vehicle leaves the depot (due to no slack and start cumul at zero so this is always 0)
    # as a result this is minimizing the maximum value in the dimension
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    # distance_dimension.SetGlobalSpanCostCoefficient(100_000)

    # # Add Capacity constraint to only allow a truck to carry one load at a _time
    # def duration_callback(from_index, to_index):
    #     """Returns the demand of the node."""
    #
    #     # Convert from routing variable Index to demands NodeIndex.
    #     from_node = manager.IndexToNode(from_index)
    #     to_node = manager.IndexToNode(to_index)
    #     return data.duration_matrix[from_node]
    # print(np.array(data.duration_matrix))

    # Set up duration matrix
    time_transit_idx = routing.RegisterTransitMatrix(data.duration_matrix)

    # Add Time constraint
    routing.AddDimension(
        time_transit_idx,
        0,  # waiting _time upper bound
        data.max_time,  # maximum duration per vehicle
        True,  # force start cumul to zero
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    time_dimension.SetGlobalSpanCostCoefficient(100_000)

    # TODO: add option for enabling or disabling this arrival window
    # print(data.num_locations)
    # print(len(data.time_windows))
    # print(data.time_windows)
    for location_idx, time_window in enumerate(data.time_windows):
        if time_window[0] == 0:
            # don't apply time window to depot
            continue
        # print(location_idx, time_window, flush=True)
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        # print("done")

    for node in range(len(data.distance_matrix)):
        if node > 3:
            # needs to be VERY high due to distances being large
            # TODO: make the early requests even harder than this to drop
            routing.AddDisjunction([manager.NodeToIndex(node)], 1_000_000_000)

    # Define request constraints (pickup and delivery constraints)
    for request in data.requests:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])

        # create a pickup and delivery request for an item
        routing.AddPickupAndDelivery(pickup_index, delivery_index)

        # each item must be picked up and delivered by the same vehicle
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index))

        # each item must be picked up before it is delivered
        # vehicle's cumulative dist at item's pickup location is at most its cumulative distance of the delivery location
        routing.solver().Add(
            distance_dimension.CumulVar(pickup_index) <= distance_dimension.CumulVar(delivery_index))

    # Add Capacity constraint to only allow a truck to carry one load at a _time
    def demand_callback(from_index):
        """Returns the demand of the node."""

        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data.demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data.vehicle_capacities,  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity")

    dimension_name = 'Count'
    routing.AddConstantDimension(
        1,  # increment by one every time
        len(data.distance_matrix),  # large enough
        True,  # set count to zero
        dimension_name)

    # define search parameters and search methods (we don't get an exact solution?)
    # https://developers.google.com/optimization/routing/routing_options
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # search_parameters.first_solution_strategy = (
    #     routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    # # search_parameters.first_solution_strategy = (
    # #     routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # search_parameters.local_search_metaheuristic = (
    #     routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    # # search_parameters.local_search_metaheuristic = (
    # #     routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
    # # # Amount of _time to search for better solutions
    # # search_parameters.time_limit.FromSeconds(60)
    search_parameters.time_limit.FromSeconds(5)

    # THESE TIME WINDOW STUFF IS BETTER BOUND BUT different than what we want (get results with and without)

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        return get_solution(data.ort, data, manager, routing, solution)
    else:
        print("Error: no feasible solution found")
        raise Exception


class OrTool(BasePolicy):
    """
    Policy of cheapest insertion
    Attributes:
        depots: a list of Depot
        customers: a list of Customer
        trucks: a list of Truck
        requests: a list of Request
    """

    def __init__(self,
                 depots: list[Depot],
                 customers: list[Customer],
                 trucks: list[Truck],
                 requests: list[Request],
                 end_time: float):
        super().__init__(
            depots,
            customers,
            trucks,
            requests,
            end_time)

        self.data: ORToolsData = _init_ortools_data(
            self.depots,
            self.customers,
            self.trucks,
            self.requests,
            end_time)
        self.init()

    def init(self):
        results = _solve(self.data)
        for truck, (count, time, distance) in zip(self.trucks, results):
            truck._deliveries_made = int((count - 1) / 2)
            truck._dock_time = time
            truck._dist_tr = distance

    def update(self, time: float):
        pass
