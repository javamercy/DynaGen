import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def compute_metrics(routes):
        lengths = [route_distance(r) for r in routes]
        return max(lengths), sum(lengths)

    def initial_solution():
        routes = [[0, 0] for _ in range(truck_count)]
        assignments = [[] for _ in range(truck_count)]
        for cust in customers:
            t = random.randint(0, truck_count-1)
            assignments[t].append(cust)
        for t in range(truck_count):
            route = assignments[t]
            random.shuffle(route)
            routes[t] = [0] + route + [0]
        return routes

    def get_neighbor(routes):
        new_routes = [r[:] for r in routes]
        # pick a random customer
        cust = random.choice(customers)
        src_idx = None
        for idx, route in enumerate(new_routes):
            if cust in route:
                src_idx = idx
                break
        if src_idx is None:
            return new_routes  # should not happen
        src_route = new_routes[src_idx]
        pos = src_route.index(cust)
        # remove customer
        src_route.pop(pos)
        # pick destination route and insertion position
        dst_idx = random.randint(0, truck_count-1)
        dst_route = new_routes[dst_idx]
        if len(dst_route) == 2:
            # only depot nodes
            ins_pos = 1
        else:
            ins_pos = random.randint(1, len(dst_route)-1)
        # insert
        dst_route.insert(ins_pos, cust)
        return new_routes

    def report_best_if_new(routes, best_max, best_total, best_routes):
        cur_max, cur_total = compute_metrics(routes)
        if cur_max < best_max or (cur_max == best_max and cur_total < best_total):
            best_max = cur_max
            best_total = cur_total
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        return best_max, best_total, best_routes

    # initial solution
    current_routes = initial_solution()
    current_max, current_total = compute_metrics(current_routes)
    best_max = current_max
    best_total = current_total
    best_routes = [r[:] for r in current_routes]
    report_best_vrp(best_routes)

    # SA parameters
    max_iter = max(1000, 100 * n)
    T0 = 0.2 * np.max(distance_matrix)
    T = T0
    cooling_rate = 0.995

    for iteration in range(max_iter):
        new_routes = get_neighbor(current_routes)
        new_max, new_total = compute_metrics(new_routes)

        # acceptance criterion
        accept = False
        if new_max < current_max or (new_max == current_max and new_total < current_total):
            accept = True
        else:
            delta = new_max - current_max
            if delta == 0:
                delta = new_total - current_total
            if delta > 0 and random.random() < math.exp(-delta / T):
                accept = True

        if accept:
            current_routes = new_routes
            current_max, current_total = new_max, new_total
            if new_max < best_max or (new_max == best_max and new_total < best_total):
                best_max, best_total, best_routes = report_best_if_new(current_routes, best_max, best_total, best_routes)

        T *= cooling_rate

    return best_routes