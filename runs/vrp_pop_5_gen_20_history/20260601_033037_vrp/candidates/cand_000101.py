import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    best_routes = None
    best_max = float('inf')
    best_total = float('inf')

    def report_best_vrp(routes):
        nonlocal best_routes, best_max, best_total
        new_max = max(route_distance(r) for r in routes)
        new_total = sum(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12 or (abs(new_max - best_max) < 1e-12 and new_total < best_total - 1e-12):
            best_max = new_max
            best_total = new_total
            best_routes = [list(r) for r in routes]

    # Construction: min-max insertion with deterministic tie-breaking (smallest customer, route, position)
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(customers)
    while unassigned:
        best_cust = None
        best_route_idx = None
        best_pos = None
        best_new_max = float('inf')
        sorted_cust = sorted(unassigned)
        for cust in sorted_cust:
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    temp_routes = [list(r) for r in routes]
                    temp_routes[r_idx] = new_route
                    new_max = max(route_distance(r) for r in temp_routes)
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_cust = cust
                        best_route_idx = r_idx
                        best_pos = pos
                    elif abs(new_max - best_new_max) < 1e-12:
                        if cust < best_cust or (cust == best_cust and r_idx < best_route_idx) or (cust == best_cust and r_idx == best_route_idx and pos < best_pos):
                            best_new_max = new_max
                            best_cust = cust
                            best_route_idx = r_idx
                            best_pos = pos
        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    report_best_vrp(routes)

    # Simulated Annealing parameters
    random.seed(42)  # deterministic randomness
    initial_temp = 100.0
    cooling_rate = 0.95
    max_iter = min(2000, n * truck_count * 3)
    temp = initial_temp
    current_routes = [list(r) for r in routes]
    current_max = best_max
    current_total = best_total

    for iteration in range(max_iter):
        # Find longest route
        dists = [route_distance(r) for r in current_routes]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], -i))
        route_max = current_routes[max_idx]
        interior = route_max[1:-1]
        if not interior:
            break

        # Randomly choose move type: relocate (0) or swap (1)
        move_type = random.randint(0, 1)
        if move_type == 0:
            # Relocate: pick random customer from longest route, random other route, random insertion position
            cust = random.choice(interior)
            other_idx = random.choice([i for i in range(truck_count) if i != max_idx])
            other_route = current_routes[other_idx]
            pos = random.randint(1, len(other_route)-1)
            new_routes = [list(r) for r in current_routes]
            new_routes[max_idx].remove(cust)
            new_routes[other_idx].insert(pos, cust)
        else:
            # Swap: pick random customer from longest route, random customer from another route
            other_idx = random.choice([i for i in range(truck_count) if i != max_idx])
            other_interior = current_routes[other_idx][1:-1]
            if not other_interior:
                continue
            cust_max = random.choice(interior)
            cust_other = random.choice(other_interior)
            new_routes = [list(r) for r in current_routes]
            idx_max = new_routes[max_idx].index(cust_max)
            idx_other = new_routes[other_idx].index(cust_other)
            new_routes[max_idx][idx_max] = cust_other
            new_routes[other_idx][idx_other] = cust_max

        # Evaluate new solution
        new_max = max(route_distance(r) for r in new_routes)
        new_total = sum(route_distance(r) for r in new_routes)

        # Acceptance criterion
        accept = False
        if new_max < current_max - 1e-12:
            accept = True
        elif abs(new_max - current_max) < 1e-12 and new_total < current_total - 1e-12:
            accept = True
        else:
            # delta = new_max - current_max (if max different) else new_total - current_total
            if abs(new_max - current_max) > 1e-12:
                delta = new_max - current_max
            else:
                delta = new_total - current_total
            if delta < 0:
                accept = True
            else:
                prob = math.exp(-delta / temp)
                if random.random() < prob:
                    accept = True

        if accept:
            current_routes = new_routes
            current_max = new_max
            current_total = new_total
            report_best_vrp(current_routes)

        # Apply intraroute 2-opt on all routes every 10 iterations
        if iteration % 10 == 0:
            for idx in range(truck_count):
                route = current_routes[idx]
                if len(route) <= 3:
                    continue
                best_route = route[:]
                best_dist = route_distance(route)
                found = False
                for a in range(1, len(route)-2):
                    for b in range(a+1, len(route)-1):
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < best_dist - 1e-12:
                            best_dist = new_dist
                            best_route = new_route
                            found = True
                    if found:
                        break
                if found:
                    current_routes[idx] = best_route
                    new_max = max(route_distance(r) for r in current_routes)
                    new_total = sum(route_distance(r) for r in current_routes)
                    if new_max < current_max - 1e-12 or (abs(new_max - current_max) < 1e-12 and new_total < current_total - 1e-12):
                        current_max = new_max
                        current_total = new_total
                        report_best_vrp(current_routes)

        # Cooling
        temp *= cooling_rate
        if temp < 1e-6:
            break

    return best_routes if best_routes is not None else routes