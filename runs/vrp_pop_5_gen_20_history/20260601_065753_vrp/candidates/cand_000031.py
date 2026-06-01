import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    def max_route_length(routes):
        return max(route_length(r) for r in routes)

    # Regret-2 construction (from parent)
    def construct():
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0.0] * truck_count
        unassigned = set(customers)
        while unassigned:
            best_cust = None
            best_diff = -1.0
            best_route = -1
            best_pos = -1
            best_max = float('inf')
            for cust in unassigned:
                first_max = None
                second_max = None
                first_route = None
                first_pos = None
                for r in range(truck_count):
                    route = routes[r]
                    for p in range(1, len(route)):
                        prev = route[p-1]
                        nxt = route[p]
                        old_edge = distance_matrix[prev, nxt]
                        new_len = lengths[r] - old_edge + distance_matrix[prev, cust] + distance_matrix[cust, nxt]
                        new_max = new_len
                        for rr in range(truck_count):
                            if rr != r and lengths[rr] > new_max:
                                new_max = lengths[rr]
                        if first_max is None or new_max < first_max:
                            second_max = first_max
                            first_max = new_max
                            first_route = r
                            first_pos = p
                        elif second_max is None or new_max < second_max:
                            second_max = new_max
                if first_max is None:
                    continue
                diff = float('inf') if second_max is None else (first_max - second_max)
                if diff > best_diff or (diff == best_diff and cust < best_cust):
                    best_diff = diff
                    best_cust = cust
                    best_route = first_route
                    best_pos = first_pos
                    best_max = first_max
            routes[best_route].insert(best_pos, best_cust)
            lengths[best_route] = route_length(routes[best_route])
            unassigned.remove(best_cust)
        return [list(r) for r in routes], lengths

    routes, lengths = construct()
    best_routes = [list(r) for r in routes]
    best_max = max(lengths)

    def report_best(routes):
        nonlocal best_max, best_routes
        m = max_route_length(routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]

    report_best(routes)

    # LNS parameters
    max_iter = n * 10
    temp_init = 1.0
    temp = temp_init
    cooling_rate = 0.999
    destroy_ratio = 0.3

    for iteration in range(max_iter):
        # Destroy: remove customers randomly
        removed_count = max(1, int(len(customers) * destroy_ratio))
        all_customers_in_routes = []
        for r in range(truck_count):
            route = routes[r]
            if len(route) > 2:
                all_customers_in_routes.extend(route[1:-1])
        if len(all_customers_in_routes) < removed_count:
            removed_count = len(all_customers_in_routes)
        removed = set(random.sample(all_customers_in_routes, removed_count))

        # Remove from routes
        new_routes = [[0, 0] for _ in range(truck_count)]
        new_lengths = [0.0] * truck_count
        unassigned = set(customers)
        for r in range(truck_count):
            route = routes[r]
            for node in route:
                if node not in removed:
                    new_routes[r].insert(len(new_routes[r])-1, node)
            new_lengths[r] = route_length(new_routes[r])
            for node in new_routes[r]:
                if node in unassigned:
                    unassigned.remove(node)
        # Now unassigned contains removed customers

        # Repair using regret-2
        while unassigned:
            best_cust = None
            best_diff = -1.0
            best_route = -1
            best_pos = -1
            best_max_new = float('inf')
            for cust in list(unassigned):
                first_max = None
                second_max = None
                first_route = None
                first_pos = None
                for r in range(truck_count):
                    route = new_routes[r]
                    for p in range(1, len(route)):
                        prev = route[p-1]
                        nxt = route[p]
                        old_edge = distance_matrix[prev, nxt]
                        new_len = new_lengths[r] - old_edge + distance_matrix[prev, cust] + distance_matrix[cust, nxt]
                        new_max = new_len
                        for rr in range(truck_count):
                            if rr != r:
                                if new_lengths[rr] > new_max:
                                    new_max = new_lengths[rr]
                        if first_max is None or new_max < first_max:
                            second_max = first_max
                            first_max = new_max
                            first_route = r
                            first_pos = p
                        elif second_max is None or new_max < second_max:
                            second_max = new_max
                if first_max is None:
                    continue
                diff = float('inf') if second_max is None else (first_max - second_max)
                if diff > best_diff or (diff == best_diff and cust < best_cust):
                    best_diff = diff
                    best_cust = cust
                    best_route = first_route
                    best_pos = first_pos
                    best_max_new = first_max
            new_routes[best_route].insert(best_pos, best_cust)
            new_lengths[best_route] = route_length(new_routes[best_route])
            unassigned.remove(best_cust)

        new_max = max(new_lengths)
        # Accept using simulated annealing (minimization)
        if new_max < best_max:
            accept = True
            report_best(new_routes)
        else:
            delta = new_max - best_max
            accept = random.random() < math.exp(-delta / temp) if temp > 0 else (new_max < best_max)
        if accept or new_max < best_max:
            routes = new_routes
            lengths = new_lengths
        # Cool down
        temp *= cooling_rate
    return best_routes