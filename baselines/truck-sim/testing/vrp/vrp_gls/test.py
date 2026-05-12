import numpy as np
import random
import time
from copy import deepcopy
from itertools import pairwise
from numba import njit, float64, int64
from numba.typed import List
from typing import List as PyList

# Constants (assuming these are defined elsewhere)
TRUCK_NUM = 3  # Example value, replace with actual constant


@njit
def route_dist(distance_mtx: np.ndarray, route: np.ndarray) -> float:
    """
    Calculate the cost of a single route, including return to depot.
    Numba-optimized version.
    """
    cost = 0.0
    for i in range(len(route) - 1):
        cost += distance_mtx[route[i], route[i + 1]]
    return cost


class Solution:
    def __init__(self, routes):
        """
        routes: list of routes, each route is a list of customer indices (excluding depot)
        """
        self.routes = routes
        self.cost = 0.0  # Will be calculated based on routes

    def calculate_cost(self, distance_mtx: np.ndarray):
        """
        Calculate the total cost based on the current routes and the distance matrix
        """
        self.cost = calculate_solution_cost(distance_mtx, self.routes)


@njit
def calculate_solution_cost(distance_mtx: np.ndarray, routes: List) -> float:
    """
    Numba-optimized solution cost calculation
    """
    total_cost = 0.0
    for route in routes:
        total_cost += route_dist(distance_mtx, route)
    return total_cost


@njit
def initial_solution(size: int, distance_mtx: np.ndarray) -> object:
    """
    Create an initial solution using the greedy insertion heuristic
    Numba-optimized version
    """
    # Prepare routes
    routes = List()
    for _ in range(TRUCK_NUM):
        routes.append(np.array([0], dtype=np.int64))

    # Prepare customer indices
    customer_indices = np.arange(1, size, dtype=np.int64)

    # Shuffle customers (Numba-compatible random shuffle)
    for i in range(len(customer_indices) - 1, 0, -1):
        j = random.randint(0, i)
        customer_indices[i], customer_indices[j] = customer_indices[j], customer_indices[i]

    # Greedy insertion
    for customer in customer_indices:
        # Find best route to insert customer
        best_route_idx = 0
        min_cost = np.inf
        for r in range(len(routes)):
            route_copy = np.concatenate((routes[r], [customer]))
            cost = route_dist(distance_mtx, route_copy)
            if cost < min_cost:
                min_cost = cost
                best_route_idx = r

        # Insert customer into best route
        routes[best_route_idx] = np.concatenate((routes[best_route_idx], [customer]))

    # Add depot at the end of each route
    for i in range(len(routes)):
        routes[i] = np.concatenate((routes[i], [0]))

    # Create solution object (note: this is a simplified representation for Numba)
    solution = type('Solution', (), {
        'routes': routes,
        'cost': calculate_solution_cost(distance_mtx, routes)
    })

    return solution


@njit
def generate_neighbors(solution, distance_mtx):
    """
    Numba-optimized neighbor generation
    """
    neighbors = List()
    routes = solution.routes

    # 2-opt within routes
    for route_idx in range(len(routes)):
        route = routes[route_idx][1:-1]  # Exclude depot

        if len(route) < 3:
            continue

        for i in range(len(route) - 1):
            for j in range(i + 2, len(route)):
                if j - i == 1:
                    continue

                # Create new route with 2-opt
                new_route = np.concatenate([
                    route[:i + 1],
                    route[i + 1:j][::-1],
                    route[j:]
                ])

                # Reconstruct full route with depots
                full_new_route = np.concatenate([[0], new_route, [0]])

                # Create new routes
                new_routes = List()
                for r_idx, r in enumerate(routes):
                    new_routes.append(full_new_route if r_idx == route_idx else r)

                # Create solution-like object
                neighbor = type('Solution', (), {
                    'routes': new_routes,
                    'cost': calculate_solution_cost(distance_mtx, new_routes)
                })

                neighbors.append(neighbor)

    return neighbors


@njit
def guided_local_search(size, distance_mtx, max_iterations=1000, running_time=10.0):
    """
    Numba-optimized Guided Local Search
    Note: This is a simplified version due to Numba's limitations
    """
    end_time = time.time() + running_time
    curr_solution = initial_solution(size, distance_mtx)
    best_solution = curr_solution

    for iteration in range(1, max_iterations + 1):
        if time.time() > end_time:
            break

        # Local search
        curr_cost = calculate_solution_cost(distance_mtx, curr_solution.routes)

        # Generate neighbors
        neighbors = generate_neighbors(curr_solution, distance_mtx)

        if len(neighbors) == 0:
            break

        # Find best neighbor
        best_neighbor_idx = 0
        best_neighbor_cost = np.inf
        for i, neighbor in enumerate(neighbors):
            neighbor_cost = calculate_solution_cost(distance_mtx, neighbor.routes)
            if neighbor_cost < best_neighbor_cost:
                best_neighbor_idx = i
                best_neighbor_cost = neighbor_cost

        best_neighbor = neighbors[best_neighbor_idx]

        # Accept if better
        if best_neighbor_cost < curr_cost:
            curr_solution = best_neighbor

            # Update best solution if actual cost is lower
            if best_neighbor_cost < calculate_solution_cost(distance_mtx, best_solution.routes):
                best_solution = best_neighbor

        if iteration % 100 == 0 or iteration == 1:
            print(f"Iteration {iteration}, Best Cost: {calculate_solution_cost(distance_mtx, best_solution.routes)}")

    return best_solution


# Example usage
def main():
    # Example initialization (replace with your actual data)
    size = 50  # number of customers
    np.random.seed(42)
    coords = np.random.rand(size + 1, 2)  # coordinates including depot
    distance_mtx = np.sqrt(((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(axis=2))

    solution = guided_local_search(size, distance_mtx)
    print("Final Solution Cost:", solution.cost)


if __name__ == "__main__":
    main()