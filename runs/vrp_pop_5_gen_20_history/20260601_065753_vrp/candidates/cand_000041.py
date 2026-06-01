import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    # Greedy construction minimizing max route distance (from parent 000033)
    def greedy_construction():
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = set(customers)
        while unassigned:
            candidates = []
            for cust in list(unassigned):
                best_inc = float('inf')
                best_truck = -1
                best_pos = -1
                for t in range(truck_count):
                    route = routes[t]
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_dist = route_distance(new_route)
                        new_max = new_dist
                        for k in range(truck_count):
                            if k != t:
                                new_max = max(new_max, route_dists[k])
                        if new_max < best_inc:
                            best_inc = new_max
                            best_truck = t
                            best_pos = pos
                candidates.append((best_inc, cust, best_truck, best_pos))
            # Deterministic tie-breaking by best_inc then cust
            candidates.sort(key=lambda x: (x[0], x[1]))
            best_inc, cust, t, pos = candidates[0]
            routes[t] = routes[t][:pos] + [cust] + routes[t][pos:]
            route_dists[t] = route_distance(routes[t])
            unassigned.remove(cust)
        return routes, route_dists

    # Insert customers greedily (tie-break: smaller truck, smaller position)
    def insert_customers(routes, route_dists, cust_list):
        for cust in cust_list:
            best_max = float('inf')
            best_truck = -1
            best_pos = -1
            for t in range(truck_count):
                route = routes[t]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_dist = route_distance(new_route)
                    new_max = new_dist
                    for k in range(truck_count):
                        if k != t:
                            new_max = max(new_max, route_dists[k])
                    if new_max < best_max:
                        best_max = new_max
                        best_truck = t
                        best_pos = pos
                    elif new_max == best_max:
                        # tie-break: smaller truck, then smaller position
                        if t < best_truck or (t == best_truck and pos < best_pos):
                            best_max = new_max
                            best_truck = t
                            best_pos = pos
            # Apply insertion
            routes[best_truck].insert(best_pos, cust)
            route_dists[best_truck] = route_distance(routes[best_truck])
        return routes, route_dists

    # Build initial solution
    routes, route_dists = greedy_construction()
    best_max = max(route_dists)
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)

    current_routes = [list(r) for r in routes]
    current_dists = list(route_dists)
    current_max = best_max

    # Simulated annealing parameters
    max_iter = max(1000, n * 10)
    T0 = best_max if best_max > 0 else 1.0
    T = T0
    cooling_rate = 1.0 / max_iter

    for iteration in range(max_iter):
        # Copy current solution
        new_routes = [list(r) for r in current_routes]
        # Destroy: remove random subset of customers
        all_cust = [node for route in new_routes for node in route if node != 0]
        if len(all_cust) == 0:
            continue
        remove_k = random.randint(1, max(1, len(all_cust) // 2))
        remove_set = set(random.sample(all_cust, remove_k))
        removed = list(remove_set)
        # Build new routes after removal
        for t in range(truck_count):
            new_routes[t] = [0] + [node for node in new_routes[t] if node != 0 and node not in remove_set] + [0]
        new_dists = [route_distance(r) for r in new_routes]
        # Shuffle removed customers for diversification
        random.shuffle(removed)
        # Repair
        new_routes, new_dists = insert_customers(new_routes, new_dists, removed)
        new_max = max(new_dists)

        # Update best if improved
        if new_max < best_max:
            best_max = new_max
            best_routes = [list(r) for r in new_routes]
            report_best_vrp(best_routes)
            current_routes = [list(r) for r in best_routes]
            current_dists = list(new_dists)
            current_max = new_max
        else:
            # Accept worse with SA probability
            delta = new_max - current_max
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_routes = [list(r) for r in new_routes]
                current_dists = list(new_dists)
                current_max = new_max
        # Cool temperature
        T = T0 * (1 - (iteration + 1) / max_iter)

    return best_routes