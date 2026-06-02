import numpy as np
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []

    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d

    def apply_relocate(routes, r_idx, pos, cust, new_r_idx, new_pos):
        routes[r_idx].pop(pos)
        routes[new_r_idx].insert(new_pos, cust)
        route_dists[r_idx] = route_distance(routes[r_idx])
        route_dists[new_r_idx] = route_distance(routes[new_r_idx])

    def apply_swap(routes, r1, pos1, r2, pos2):
        cust1 = routes[r1][pos1]
        cust2 = routes[r2][pos2]
        routes[r1][pos1] = cust2
        routes[r2][pos2] = cust1
        route_dists[r1] = route_distance(routes[r1])
        route_dists[r2] = route_distance(routes[r2])

    def apply_cross(routes, r1, pos1, r2, pos2):
        tail1 = routes[r1][pos1+1:]
        tail2 = routes[r2][pos2+1:]
        new_route1 = routes[r1][:pos1+1] + tail2
        new_route2 = routes[r2][:pos2+1] + tail1
        routes[r1] = new_route1
        routes[r2] = new_route2
        route_dists[r1] = route_distance(new_route1)
        route_dists[r2] = route_distance(new_route2)

    def construct_and_search(order):
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = list(order)
        route_dists = [route_distance(r) for r in routes]
        for cust in unassigned:
            best_new_max = float('inf')
            best_route_idx = -1
            best_pos = -1
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    succ = route[pos]
                    increase = distance_matrix[prev][cust] + distance_matrix[cust][succ] - distance_matrix[prev][succ]
                    new_route_dist = route_dists[r_idx] + increase
                    if new_route_dist < best_new_max:
                        new_max = new_route_dist
                        for other_idx, d in enumerate(route_dists):
                            if other_idx != r_idx and d > new_max:
                                new_max = d
                        if new_max < best_new_max or (new_max == best_new_max and r_idx < best_route_idx):
                            best_new_max = new_max
                            best_route_idx = r_idx
                            best_pos = pos
                    elif new_route_dist == best_new_max:
                        if r_idx < best_route_idx:
                            best_new_max = new_max
                            best_route_idx = r_idx
                            best_pos = pos
            route = routes[best_route_idx]
            route.insert(best_pos, cust)
            route_dists[best_route_idx] = route_distance(route)
        # local search
        max_iter = min(n * 5, 200)
        for _ in range(max_iter):
            current_max = max(route_dists)
            max_routes = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
            improved = False
            # relocate
            for r_idx in max_routes:
                route = routes[r_idx]
                for pos in range(1, len(route)-1):
                    cust = route[pos]
                    prev = route[pos-1]
                    succ = route[pos+1]
                    removal_change = distance_matrix[prev][succ] - (distance_matrix[prev][cust] + distance_matrix[cust][succ])
                    new_dist_r = route_dists[r_idx] + removal_change
                    for other_idx in range(truck_count):
                        if other_idx == r_idx:
                            continue
                        other_route = routes[other_idx]
                        for insert_pos in range(1, len(other_route)):
                            prev2 = other_route[insert_pos-1]
                            succ2 = other_route[insert_pos]
                            insertion_change = distance_matrix[prev2][cust] + distance_matrix[cust][succ2] - distance_matrix[prev2][succ2]
                            new_dist_other = route_dists[other_idx] + insertion_change
                            new_max = max(new_dist_r, new_dist_other)
                            for idx, d in enumerate(route_dists):
                                if idx != r_idx and idx != other_idx and d > new_max:
                                    new_max = d
                            if new_max < current_max - 1e-12:
                                apply_relocate(routes, r_idx, pos, cust, other_idx, insert_pos)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                yield routes, route_dists
                continue
            # swap
            for r1 in max_routes:
                route1 = routes[r1]
                for pos1 in range(1, len(route1)-1):
                    cust1 = route1[pos1]
                    for r2 in range(truck_count):
                        if r2 == r1:
                            continue
                        route2 = routes[r2]
                        for pos2 in range(1, len(route2)-1):
                            cust2 = route2[pos2]
                            prev1 = route1[pos1-1]
                            succ1 = route1[pos1+1]
                            remove1_change = distance_matrix[prev1][succ1] - (distance_matrix[prev1][cust1] + distance_matrix[cust1][succ1])
                            prev2 = route2[pos2-1]
                            succ2 = route2[pos2+1]
                            remove2_change = distance_matrix[prev2][succ2] - (distance_matrix[prev2][cust2] + distance_matrix[cust2][succ2])
                            insert1_change = distance_matrix[prev1][cust2] + distance_matrix[cust2][succ1] - distance_matrix[prev1][succ1]
                            insert2_change = distance_matrix[prev2][cust1] + distance_matrix[cust1][succ2] - distance_matrix[prev2][succ2]
                            new_dist_r1 = route_dists[r1] + remove1_change + insert1_change
                            new_dist_r2 = route_dists[r2] + remove2_change + insert2_change
                            new_max = max(new_dist_r1, new_dist_r2)
                            for idx, d in enumerate(route_dists):
                                if idx != r1 and idx != r2 and d > new_max:
                                    new_max = d
                            if new_max < current_max - 1e-12:
                                apply_swap(routes, r1, pos1, r2, pos2)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                yield routes, route_dists
                continue
            # 2-opt
            for r_idx in max_routes:
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old_edges = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                        new_edges = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                        change = new_edges - old_edges
                        new_dist = route_dists[r_idx] + change
                        if new_dist < current_max - 1e-12:
                            route[i:j+1] = reversed(route[i:j+1])
                            route_dists[r_idx] = route_distance(route)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                yield routes, route_dists
                continue
            # cross
            for r1 in max_routes:
                route1 = routes[r1]
                for pos1 in range(0, len(route1)-1):
                    if pos1 == len(route1)-1:
                        continue
                    for r2 in range(truck_count):
                        if r2 == r1:
                            continue
                        route2 = routes[r2]
                        for pos2 in range(0, len(route2)-1):
                            if pos2 == len(route2)-1:
                                continue
                            delta1 = distance_matrix[route1[pos1]][route2[pos2+1]] - distance_matrix[route1[pos1]][route1[pos1+1]]
                            delta2 = distance_matrix[route2[pos2]][route1[pos1+1]] - distance_matrix[route2[pos2]][route2[pos2+1]]
                            new_dist_r1 = route_dists[r1] + delta1
                            new_dist_r2 = route_dists[r2] + delta2
                            new_max = max(new_dist_r1, new_dist_r2)
                            for idx, d in enumerate(route_dists):
                                if idx != r1 and idx != r2 and d > new_max:
                                    new_max = d
                            if new_max < current_max - 1e-12:
                                apply_cross(routes, r1, pos1, r2, pos2)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                yield routes, route_dists
                continue
            break
        yield routes, route_dists

    best_routes = None
    best_max = float('inf')

    # Restart 1: descending distance from depot
    order1 = list(range(1, n))
    order1.sort(key=lambda c: (-distance_matrix[0][c], c))
    for routes, route_dists in construct_and_search(order1):
        current_max = max(route_dists)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)

    # Restart 2: ascending distance from depot
    order2 = list(range(1, n))
    order2.sort(key=lambda c: (distance_matrix[0][c], c))
    for routes, route_dists in construct_and_search(order2):
        current_max = max(route_dists)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)

    # Restart 3: natural order (1..n-1)
    order3 = list(range(1, n))
    for routes, route_dists in construct_and_search(order3):
        current_max = max(route_dists)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)

    if best_routes is None:
        # fallback: empty routes
        best_routes = [[0, 0] for _ in range(truck_count)]
    return best_routes