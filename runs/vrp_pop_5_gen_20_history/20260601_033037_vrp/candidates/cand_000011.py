import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    num_customers = len(customers)
    if num_customers == 0:
        return [[0,0] for _ in range(truck_count)]

    # Pheromone matrix: edges i->j (i,j from 0 to n-1)
    tau0 = 1.0 / (n - 1)
    pheromone = np.full((n, n), tau0)
    # Parameters
    alpha = 1.0
    beta = 2.0
    rho = 0.1
    Q = 1.0
    max_iter = min(50, n * 2)
    num_ants = max(10, truck_count * 2)

    best_routes = None
    best_max = float('inf')

    def route_distance(route):
        # route is list of customers without depots
        if not route:
            return 0.0
        d = distance_matrix[0, route[0]] + distance_matrix[route[-1], 0]
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def compute_max(routes):
        return max(route_distance(r) for r in routes)

    def insert_customer(route, cust):
        # insert cust into route at position that minimizes insertion cost, return new route and delta
        best_delta = float('inf')
        best_pos = None
        if not route:
            delta = distance_matrix[0, cust] + distance_matrix[cust, 0]
            best_delta = delta
            best_pos = 0
        else:
            for pos in range(len(route) + 1):
                if pos == 0:
                    prev = 0
                    nxt = route[0]
                elif pos == len(route):
                    prev = route[-1]
                    nxt = 0
                else:
                    prev = route[pos-1]
                    nxt = route[pos]
                delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                if delta < best_delta:
                    best_delta = delta
                    best_pos = pos
        new_route = route[:best_pos] + [cust] + route[best_pos:]
        return new_route, best_delta

    # Main ACO loop
    for iteration in range(max_iter):
        for ant in range(num_ants):
            # Build solution
            routes = [[] for _ in range(truck_count)]
            # Random order of customers
            unassigned = list(customers)
            random.shuffle(unassigned)
            for cust in unassigned:
                # Compute attractiveness for each truck
                attractiveness = []
                for t_idx, route in enumerate(routes):
                    # compute best insertion delta and position
                    new_route, delta = insert_customer(route, cust)
                    # heuristic = 1/(1+delta) to avoid division by zero
                    eta = 1.0 / (1.0 + delta)
                    # pheromone: average pheromone on edges from last customer in route to cust and from cust to next?
                    # simplified: use pheromone on edge from previous node (last or depot) to cust
                    if not route:
                        prev_node = 0
                    else:
                        prev_node = route[-1]
                    tau = pheromone[prev_node, cust]
                    # attractiveness
                    a = (tau ** alpha) * (eta ** beta)
                    attractiveness.append(a)
                # Normalize probabilities
                total = sum(attractiveness)
                if total > 0:
                    probs = [a / total for a in attractiveness]
                else:
                    probs = [1.0 / truck_count] * truck_count
                # Choose truck via roulette wheel (deterministic tie-breaking by index)
                r = random.random()
                cum = 0.0
                chosen_truck = truck_count - 1
                for t_idx, p in enumerate(probs):
                    cum += p
                    if r < cum:
                        chosen_truck = t_idx
                        break
                # Insert customer in chosen truck at best position
                route = routes[chosen_truck]
                new_route, _ = insert_customer(route, cust)
                routes[chosen_truck] = new_route
            # Evaluate solution
            current_max = compute_max(routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = [list(r) for r in routes]
                # Report best
                full_routes = [[0] + r + [0] for r in best_routes]
                # report_best_vrp is expected to be defined externally
                report_best_vrp(full_routes)
        # Pheromone update: evaporate
        pheromone *= (1 - rho)
        # Deposit on edges of best solution found so far
        if best_routes is not None:
            for route in best_routes:
                if not route:
                    continue
                # edges: depot -> first, last -> depot, between customers
                edges = []
                edges.append((0, route[0]))
                for i in range(len(route) - 1):
                    edges.append((route[i], route[i+1]))
                edges.append((route[-1], 0))
                delta_tau = Q / best_max
                for (i, j) in edges:
                    pheromone[i, j] += delta_tau
        # Simple local search on best solution (optional) - 2-opt on each route
        if best_routes is not None:
            improved = True
            while improved:
                improved = False
                for t_idx, route in enumerate(best_routes):
                    if len(route) < 3:
                        continue
                    best_route = route[:]
                    best_dist = route_distance(route)
                    for a in range(len(route) - 1):
                        for b in range(a+2, len(route) + 1):
                            if b - a < 2:
                                continue
                            new_route = route[:a] + route[a:b][::-1] + route[b:]
                            new_dist = route_distance(new_route)
                            if new_dist < best_dist:
                                best_dist = new_dist
                                best_route = new_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        best_routes[t_idx] = best_route
                        # update best_max
                        new_max = compute_max(best_routes)
                        if new_max < best_max:
                            best_max = new_max
                            full_routes = [[0] + r + [0] for r in best_routes]
                            report_best_vrp(full_routes)
                        break
                # end while
    # Ensure exactly truck_count routes
    full_routes = [[0] + r + [0] for r in best_routes]
    while len(full_routes) < truck_count:
        full_routes.append([0,0])
    return full_routes