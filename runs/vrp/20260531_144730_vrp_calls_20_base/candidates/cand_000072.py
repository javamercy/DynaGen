import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    random.seed(0)

    # Initialization: regret construction
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = list(range(1, n))

    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def best_insertion(c, routes, route_dists):
        best = (float('inf'), -1, -1)
        second = (float('inf'), -1, -1)
        for r_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            other_max = 0.0
            for j, d in enumerate(route_dists):
                if j != r_idx and d > other_max:
                    other_max = d
            for pos in range(1, len(route)):
                pred = route[pos-1]
                succ = route[pos]
                new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                new_max = max(other_max, new_dist)
                if new_max < best[0]:
                    best, second = (new_max, r_idx, pos), best
                elif new_max < second[0]:
                    second = (new_max, r_idx, pos)
        return best[0], best[1], best[2], second[0]

    while unassigned:
        bests = []
        for c in unassigned:
            best_new_max, best_route, best_pos, second_new_max = best_insertion(c, routes, route_dists)
            if best_route == -1:
                continue
            regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
            bests.append((-regret, c, best_route, best_pos, best_new_max))
        bests.sort(key=lambda x: (x[0], x[1]))
        _, c, best_route, best_pos, new_max = bests[0]
        route = routes[best_route]
        route.insert(best_pos, c)
        route_dists[best_route] = route_dist(route)
        unassigned.remove(c)
        report_best_vrp(routes)

    best_routes = [route[:] for route in routes]
    best_max = max(route_dists)
    current_routes = [route[:] for route in routes]
    current_dists = route_dists[:]

    # LNS parameters
    max_iter = n * 10
    for _ in range(max_iter):
        # Destroy: worst removal (remove customers with highest detour cost)
        num_remove = random.randint(max(1, (n-1)//10), max(1, (n-1)*4//10))
        # Compute removal cost for each customer
        removal_costs = []
        for c in range(1, n):
            for r_idx, route in enumerate(current_routes):
                if c in route:
                    pos = route.index(c)
                    pred = route[pos-1]
                    succ = route[pos+1]
                    cost = distance_matrix[pred, c] + distance_matrix[c, succ] - distance_matrix[pred, succ]
                    removal_costs.append((cost, c, r_idx, pos, pred, succ))
                    break
        # Sort by cost descending and choose a random subset from the worst half
        removal_costs.sort(key=lambda x: -x[0])
        num_worst = max(1, len(removal_costs)//2)
        candidates = removal_costs[:num_worst]
        random.shuffle(candidates)
        to_remove = [c[1] for c in candidates[:num_remove]]

        # Remove them from temp solution
        temp_routes = [route[:] for route in current_routes]
        temp_dists = current_dists[:]
        for c in to_remove:
            found = False
            for r_idx in range(truck_count):
                if c in temp_routes[r_idx]:
                    pos = temp_routes[r_idx].index(c)
                    pred = temp_routes[r_idx][pos-1]
                    succ = temp_routes[r_idx][pos+1]
                    temp_dists[r_idx] += distance_matrix[pred, succ] - distance_matrix[pred, c] - distance_matrix[c, succ]
                    temp_routes[r_idx].pop(pos)
                    found = True
                    break
            if not found:
                # Should not happen, but safety
                pass

        # Repair: reinsert removed customers using regret (same as construction)
        unassigned = to_remove[:]
        while unassigned:
            bests = []
            for c in unassigned:
                best_new_max, best_route, best_pos, second_new_max = best_insertion(c, temp_routes, temp_dists)
                if best_route == -1:
                    continue
                regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
                bests.append((-regret, c, best_route, best_pos, best_new_max))
            bests.sort(key=lambda x: (x[0], x[1]))
            _, c, best_route, best_pos, new_max = bests[0]
            route = temp_routes[best_route]
            route.insert(best_pos, c)
            temp_dists[best_route] = route_dist(route)
            unassigned.remove(c)

        # Evaluate
        new_max = max(temp_dists)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [route[:] for route in temp_routes]
            report_best_vrp(best_routes)
            current_routes = [route[:] for route in temp_routes]
            current_dists = temp_dists[:]
        # else: keep current solution unchanged

    return best_routes