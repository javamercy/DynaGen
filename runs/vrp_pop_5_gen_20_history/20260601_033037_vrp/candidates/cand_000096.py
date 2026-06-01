import numpy as np
import random
import collections

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(42)
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

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]

    # Construction: min-max insertion (from parent cand_000084)
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
                        if cust < best_cust or (cust == best_cust and r_idx < best_route_idx) or \
                           (cust == best_cust and r_idx == best_route_idx and pos < best_pos):
                            best_new_max = new_max
                            best_cust = cust
                            best_route_idx = r_idx
                            best_pos = pos
        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    report_best_vrp(routes)

    # Simulated annealing parameters
    max_iter = min(2000, n * truck_count * 3)
    T = 100.0
    cooling_rate = 0.99
    stagnation_limit = max(100, n * 2)
    no_improve = 0

    for iteration in range(max_iter):
        # Find longest route
        dists = [route_distance(r) for r in routes]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        route_max = routes[max_idx]
        interior = route_max[1:-1]
        if not interior:
            break

        # Collect all valid moves (relocate and swap) into a list
        moves = []
        # Relocate moves from max route to other routes
        for cust in interior:
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for pos in range(1, len(other_route)):
                    new_routes = [list(r) for r in routes]
                    new_routes[max_idx].remove(cust)
                    new_routes[other_idx].insert(pos, cust)
                    new_max = max(route_distance(r) for r in new_routes)
                    moves.append(('relocate', cust, max_idx, other_idx, pos, new_routes, new_max))
        # Swap moves between max route and other routes
        for other_idx in range(truck_count):
            if other_idx == max_idx:
                continue
            other_interior = routes[other_idx][1:-1]
            if not other_interior:
                continue
            for cust_max in interior:
                for cust_other in other_interior:
                    new_routes = [list(r) for r in routes]
                    idx_max = new_routes[max_idx].index(cust_max)
                    idx_other = new_routes[other_idx].index(cust_other)
                    new_routes[max_idx][idx_max] = cust_other
                    new_routes[other_idx][idx_other] = cust_max
                    new_max = max(route_distance(r) for r in new_routes)
                    moves.append(('swap', cust_max, max_idx, cust_other, other_idx, new_routes, new_max))

        if not moves:
            break

        # Randomly pick a move
        move = random.choice(moves)
        if move[0] == 'relocate':
            _, cust, from_idx, to_idx, pos, new_routes, new_max = move
        else:
            _, cust_max, max_idx, cust_other, other_idx, new_routes, new_max = move

        current_max = max(route_distance(r) for r in routes)
        delta = new_max - current_max

        # Acceptance criterion
        if delta < 0 or random.random() < np.exp(-delta / T) if T > 0 else False:
            routes = new_routes
            if new_max < best_max - 1e-12:
                report_best_vrp(routes)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        # Intra-route 2-opt on all routes
        for idx in range(truck_count):
            route = routes[idx]
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
                        break
                if found:
                    break
            if found:
                routes[idx] = best_route
                new_max = max(route_distance(r) for r in routes)
                if new_max < best_max - 1e-12:
                    report_best_vrp(routes)

        # Cooling
        T *= cooling_rate

        # Stagnation handling: reheat and perturb
        if no_improve >= stagnation_limit:
            T = 100.0  # reset temperature
            # Perturb: swap two customers from different routes at random
            if truck_count >= 2:
                r1 = random.randint(0, truck_count-1)
                r2 = random.randint(0, truck_count-1)
                while r2 == r1:
                    r2 = random.randint(0, truck_count-1)
                route1 = routes[r1]
                route2 = routes[r2]
                if len(route1) > 2 and len(route2) > 2:
                    i1 = random.randint(1, len(route1)-2)
                    i2 = random.randint(1, len(route2)-2)
                    cust1 = route1[i1]
                    cust2 = route2[i2]
                    routes[r1][i1] = cust2
                    routes[r2][i2] = cust1
                    new_max = max(route_distance(r) for r in routes)
                    if new_max < best_max - 1e-12:
                        report_best_vrp(routes)
            no_improve = 0

    return best_routes if best_routes is not None else routes