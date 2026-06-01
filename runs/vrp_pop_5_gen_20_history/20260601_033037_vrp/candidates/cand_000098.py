import numpy as np
import collections

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

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]

    # Regret-2 construction
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(customers)
    while unassigned:
        best_cust = None
        best_route_idx = None
        best_pos = None
        best_regret = -float('inf')
        best_second_max = float('inf')
        # For each unassigned customer, compute best and second best insertion
        for cust in sorted(unassigned):
            insertion_costs = []
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    temp_routes = [list(r) for r in routes]
                    temp_routes[r_idx] = new_route
                    new_max = max(route_distance(r) for r in temp_routes)
                    insertion_costs.append((new_max, r_idx, pos, cust))
            # Sort by new_max (cost) ascending
            insertion_costs.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
            best = insertion_costs[0]
            second = insertion_costs[1] if len(insertion_costs) > 1 else (float('inf'), -1, -1, cust)
            regret = second[0] - best[0]
            # Tie-break: larger regret, then smaller customer id
            if regret > best_regret + 1e-12:
                best_regret = regret
                best_cust = cust
                best_route_idx = best[1]
                best_pos = best[2]
            elif abs(regret - best_regret) < 1e-12:
                if cust < best_cust:
                    best_regret = regret
                    best_cust = cust
                    best_route_idx = best[1]
                    best_pos = best[2]
        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    report_best_vrp(routes)

    # Tabu search parameters
    max_iter = min(1000, n * truck_count * 2)
    tenure = 10
    tabu_list = collections.deque(maxlen=tenure)

    for iteration in range(max_iter):
        # Find longest route
        dists = [route_distance(r) for r in routes]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        route_max = routes[max_idx]
        interior = route_max[1:-1]
        if not interior:
            break

        best_move = None
        best_new_max = float('inf')

        # Relocate moves from max route to other routes (non-tabu)
        for cust in interior:
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                move_key = (cust, max_idx, other_idx)
                if move_key in tabu_list:
                    continue
                other_route = routes[other_idx]
                for pos in range(1, len(other_route)):
                    new_routes = [list(r) for r in routes]
                    new_routes[max_idx].remove(cust)
                    new_routes[other_idx].insert(pos, cust)
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_move = ('relocate', cust, max_idx, other_idx, pos, new_routes)
                    elif abs(new_max - best_new_max) < 1e-12:
                        if cust < best_move[1] if best_move else True:
                            best_new_max = new_max
                            best_move = ('relocate', cust, max_idx, other_idx, pos, new_routes)

        # Apply best move if found
        if best_move is not None:
            _, cust, from_idx, to_idx, pos, new_routes = best_move
            move_key = (cust, from_idx, to_idx)
            # Aspiration: accept if improving best overall
            if best_new_max < best_max - 1e-12:
                tabu_list.append(move_key)
                routes = [list(r) for r in new_routes]
                report_best_vrp(routes)
            else:
                tabu_list.append(move_key)
                routes = [list(r) for r in new_routes]
        else:
            pass

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

    return best_routes if best_routes is not None else routes