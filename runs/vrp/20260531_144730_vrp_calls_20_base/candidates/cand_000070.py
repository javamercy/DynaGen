import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    random.seed(0)

    def route_dist(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def eval_max(routes):
        return max(route_dist(r) for r in routes)

    # Regret insertion heuristic
    def best_insertion(c, routes, dists):
        best = (float('inf'), -1, -1)
        second = (float('inf'), -1, -1)
        for r_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            other_max = max(dists[j] for j in range(len(dists)) if j != r_idx) if len(dists) > 1 else 0.0
            for pos in range(1, len(route)):
                pred, succ = route[pos-1], route[pos]
                new_dist = dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                new_max = max(other_max, new_dist)
                if new_max < best[0]:
                    best, second = (new_max, r_idx, pos), best
                elif new_max < second[0]:
                    second = (new_max, r_idx, pos)
        return best[0], best[1], best[2], second[0]

    # Construction
    routes = [[0, 0] for _ in range(truck_count)]
    dists = [0.0] * truck_count
    unassigned = list(range(1, n))
    while unassigned:
        candidates = []
        for c in unassigned:
            best_val, best_route, best_pos, second_val = best_insertion(c, routes, dists)
            if best_route == -1:
                continue
            regret = second_val - best_val if second_val != float('inf') else float('inf')
            candidates.append((-regret, c, best_route, best_pos, best_val))
        candidates.sort(key=lambda x: (x[0], x[1]))
        _, c, r, p, _ = candidates[0]
        routes[r].insert(p, c)
        dists[r] = route_dist(routes[r])
        unassigned.remove(c)

    best_routes = [r[:] for r in routes]
    best_max = eval_max(best_routes)

    def improve(routes, dists):
        # Intra-route 2-opt
        for idx in range(truck_count):
            improved = True
            while improved:
                improved = False
                route = routes[idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                        new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        if new < old - 1e-12:
                            route[i:j+1] = reversed(route[i:j+1])
                            improved = True
                            dists[idx] = route_dist(route)
                            break
                    if improved:
                        break
        # Inter-route relocate (best-improvement, focuses on longest route)
        improved_overall = True
        while improved_overall:
            improved_overall = False
            max_dist = max(dists)
            max_idx = dists.index(max_dist)
            best_move = None
            best_new_max = max_dist
            route_max = routes[max_idx]
            for i in range(1, len(route_max)-1):
                c = route_max[i]
                pred, succ = route_max[i-1], route_max[i+1]
                new_max_dist = dists[max_idx] - distance_matrix[pred, c] - distance_matrix[c, succ] + distance_matrix[pred, succ]
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    for pos in range(1, len(other_route)):
                        pred_o, succ_o = other_route[pos-1], other_route[pos]
                        new_other = dists[other_idx] - distance_matrix[pred_o, succ_o] + distance_matrix[pred_o, c] + distance_matrix[c, succ_o]
                        other_max = max(dists[j] for j in range(truck_count) if j not in (max_idx, other_idx)) if truck_count > 2 else 0.0
                        new_overall = max(other_max, new_max_dist, new_other)
                        if new_overall < best_new_max - 1e-12:
                            best_new_max = new_overall
                            best_move = (i, other_idx, pos, new_max_dist, new_other)
            if best_move is not None:
                i, other_idx, pos, new_max_dist, new_other = best_move
                c = route_max.pop(i)
                routes[other_idx].insert(pos, c)
                dists[max_idx] = new_max_dist
                dists[other_idx] = new_other
                improved_overall = True
                # re-apply 2-opt on affected routes
                for idx in (max_idx, other_idx):
                    improved = True
                    while improved:
                        improved = False
                        route = routes[idx]
                        for a in range(1, len(route)-2):
                            for b in range(a+1, len(route)-1):
                                old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                if new < old - 1e-12:
                                    route[a:b+1] = reversed(route[a:b+1])
                                    improved = True
                                    dists[idx] = route_dist(route)
                                    break
                            if improved:
                                break
        return routes, dists

    # Initial improvement
    routes, dists = improve(routes, dists)
    cur_max = max(dists)
    if cur_max < best_max - 1e-12:
        best_max = cur_max
        best_routes = [r[:] for r in routes]

    # Hybrid ILS-LNS: perturb best solution with LNS destroy-repair, then improve
    outer_iter = max(1, n // 10)
    for _ in range(outer_iter):
        temp_routes = [r[:] for r in best_routes]
        temp_dists = [route_dist(r) for r in temp_routes]
        # Destroy: remove random subset (10% to 30% of customers)
        num_remove = random.randint(max(1, (n-1)//10), max(1, (n-1)*3//10))
        customers = list(range(1, n))
        random.shuffle(customers)
        to_remove = customers[:num_remove]
        for c in to_remove:
            for idx, route in enumerate(temp_routes):
                if c in route:
                    pos = route.index(c)
                    pred, succ = route[pos-1], route[pos+1]
                    temp_dists[idx] += distance_matrix[pred, succ] - distance_matrix[pred, c] - distance_matrix[c, succ]
                    route.pop(pos)
                    break
        # Repair with regret
        unassigned = to_remove[:]
        while unassigned:
            candidates = []
            for c in unassigned:
                best_val, best_route, best_pos, second_val = best_insertion(c, temp_routes, temp_dists)
                if best_route == -1:
                    continue
                regret = second_val - best_val if second_val != float('inf') else float('inf')
                candidates.append((-regret, c, best_route, best_pos, best_val))
            candidates.sort(key=lambda x: (x[0], x[1]))
            _, c, r, p, _ = candidates[0]
            temp_routes[r].insert(p, c)
            temp_dists[r] = route_dist(temp_routes[r])
            unassigned.remove(c)
        # Improve
        temp_routes, temp_dists = improve(temp_routes, temp_dists)
        new_max = max(temp_dists)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [r[:] for r in temp_routes]
            report_best_vrp(best_routes)
    return best_routes