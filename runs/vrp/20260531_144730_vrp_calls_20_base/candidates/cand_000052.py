import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    random.seed(0)
    
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

    def regret_construction():
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = list(range(1, n))
        while unassigned:
            bests = []
            for c in unassigned:
                best_new_max, best_route, best_pos, second_new_max = best_insertion(c, routes, route_dists)
                if best_route == -1:
                    continue
                regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
                bests.append((-regret, c, best_route, best_pos, best_new_max))
            bests.sort(key=lambda x: (x[0], x[1]))
            _, c, best_route, best_pos, _ = bests[0]
            routes[best_route].insert(best_pos, c)
            route_dists[best_route] = route_dist(routes[best_route])
            unassigned.remove(c)
            report_best_vrp(routes)
        return routes, route_dists

    def intra_2opt(routes, route_dists, affected):
        for r_idx in affected:
            improved = True
            while improved:
                improved = False
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                        new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        if new < old - 1e-12:
                            route[i:j+1] = reversed(route[i:j+1])
                            improved = True
                            route_dists[r_idx] = route_dist(route)
                            break
                    if improved:
                        break
        return routes, route_dists

    def find_best_relocate(routes, route_dists):
        best_move = None
        best_new_max = max(route_dists)
        max_idx = route_dists.index(best_new_max)
        route_max = routes[max_idx]
        for i in range(1, len(route_max)-1):
            c = route_max[i]
            pred = route_max[i-1]
            succ = route_max[i+1]
            new_from_dist = route_dists[max_idx] - distance_matrix[pred, c] - distance_matrix[c, succ] + distance_matrix[pred, succ]
            for to_idx in range(truck_count):
                if to_idx == max_idx:
                    continue
                to_route = routes[to_idx]
                for pos in range(1, len(to_route)):
                    pred_o = to_route[pos-1]
                    succ_o = to_route[pos]
                    new_to_dist = route_dists[to_idx] - distance_matrix[pred_o, succ_o] + distance_matrix[pred_o, c] + distance_matrix[c, succ_o]
                    other_max = 0.0
                    for k, d in enumerate(route_dists):
                        if k != max_idx and k != to_idx and d > other_max:
                            other_max = d
                    new_overall = max(other_max, new_from_dist, new_to_dist)
                    if new_overall < best_new_max - 1e-12:
                        best_new_max = new_overall
                        best_move = (max_idx, i, to_idx, pos, new_from_dist, new_to_dist)
        return best_move, best_new_max

    def find_best_swap(routes, route_dists):
        best_move = None
        best_new_max = max(route_dists)
        max_idx = route_dists.index(best_new_max)
        route_max = routes[max_idx]
        for i in range(1, len(route_max)-1):
            c1 = route_max[i]
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for j in range(1, len(other_route)-1):
                    c2 = other_route[j]
                    pred1 = route_max[i-1]
                    succ1 = route_max[i+1]
                    new_dist_max = route_dists[max_idx] - distance_matrix[pred1, c1] - distance_matrix[c1, succ1] + distance_matrix[pred1, c2] + distance_matrix[c2, succ1]
                    pred2 = other_route[j-1]
                    succ2 = other_route[j+1]
                    new_dist_other = route_dists[other_idx] - distance_matrix[pred2, c2] - distance_matrix[c2, succ2] + distance_matrix[pred2, c1] + distance_matrix[c1, succ2]
                    other_max = 0.0
                    for k, d in enumerate(route_dists):
                        if k != max_idx and k != other_idx and d > other_max:
                            other_max = d
                    new_overall = max(other_max, new_dist_max, new_dist_other)
                    if new_overall < best_new_max - 1e-12:
                        best_new_max = new_overall
                        best_move = (max_idx, i, other_idx, j, new_dist_max, new_dist_other)
        return best_move, best_new_max

    def find_best_2opt_star(routes, route_dists):
        best_move = None
        best_new_max = max(route_dists)
        max_idx = route_dists.index(best_new_max)
        route_max = routes[max_idx]
        for other_idx in range(truck_count):
            if other_idx == max_idx:
                continue
            other_route = routes[other_idx]
            for i in range(1, len(route_max)-1):
                for j in range(1, len(other_route)-1):
                    if route_max[-1] != 0 or other_route[-1] != 0:
                        continue
                    old1 = distance_matrix[route_max[i], route_max[i+1]]
                    old2 = distance_matrix[other_route[j], other_route[j+1]]
                    new1 = distance_matrix[route_max[i], other_route[j+1]]
                    new2 = distance_matrix[other_route[j], route_max[i+1]]
                    new_dist_max = route_dists[max_idx] - old1 + new1
                    new_dist_other = route_dists[other_idx] - old2 + new2
                    other_max = 0.0
                    for k, d in enumerate(route_dists):
                        if k != max_idx and k != other_idx and d > other_max:
                            other_max = d
                    new_overall = max(other_max, new_dist_max, new_dist_other)
                    if new_overall < best_new_max - 1e-12:
                        best_new_max = new_overall
                        best_move = (max_idx, i, other_idx, j, route_max[i+1:], other_route[j+1:], new_dist_max, new_dist_other)
        return best_move, best_new_max

    def apply_move(routes, route_dists, move_type, move):
        if move_type == 'relocate':
            from_idx, i, to_idx, pos, new_from, new_to = move
            c = routes[from_idx].pop(i)
            routes[to_idx].insert(pos, c)
            route_dists[from_idx] = new_from
            route_dists[to_idx] = new_to
        elif move_type == 'swap':
            max_idx, i, other_idx, j, new_max, new_other = move
            route_max = routes[max_idx]
            other_route = routes[other_idx]
            c1 = route_max[i]
            c2 = other_route[j]
            route_max[i] = c2
            other_route[j] = c1
            route_dists[max_idx] = new_max
            route_dists[other_idx] = new_other
        elif move_type == '2opt_star':
            max_idx, i, other_idx, j, suffix_max, suffix_other, new_max, new_other = move
            route_max = routes[max_idx]
            other_route = routes[other_idx]
            new_route_max = route_max[:i+1] + suffix_other
            new_route_other = other_route[:j+1] + suffix_max
            routes[max_idx] = new_route_max
            routes[other_idx] = new_route_other
            route_dists[max_idx] = new_max
            route_dists[other_idx] = new_other
        return routes, route_dists

    # Initial construction
    routes, route_dists = regret_construction()
    routes, route_dists = intra_2opt(routes, route_dists, list(range(truck_count)))
    best_routes = [route[:] for route in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)

    # Main loop: VND + Shake
    max_restarts = 5
    max_vnd_iter = n * truck_count
    shake_iterations = n
    for restart in range(max_restarts):
        # VND: cycle through neighborhoods
        for vnd_iter in range(max_vnd_iter):
            improved = False
            # Relocate
            move, new_max = find_best_relocate(routes, route_dists)
            if move is not None:
                routes, route_dists = apply_move(routes, route_dists, 'relocate', move)
                routes, route_dists = intra_2opt(routes, route_dists, [move[0], move[2]])
                improved = True
                cur_max = max(route_dists)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [route[:] for route in routes]
                    report_best_vrp(best_routes)
            # Swap
            if not improved:
                move, new_max = find_best_swap(routes, route_dists)
                if move is not None:
                    routes, route_dists = apply_move(routes, route_dists, 'swap', move)
                    routes, route_dists = intra_2opt(routes, route_dists, [move[0], move[2]])
                    improved = True
                    cur_max = max(route_dists)
                    if cur_max < best_max - 1e-12:
                        best_max = cur_max
                        best_routes = [route[:] for route in routes]
                        report_best_vrp(best_routes)
            # 2-opt*
            if not improved:
                move, new_max = find_best_2opt_star(routes, route_dists)
                if move is not None:
                    routes, route_dists = apply_move(routes, route_dists, '2opt_star', move)
                    routes, route_dists = intra_2opt(routes, route_dists, [move[0], move[2]])
                    improved = True
                    cur_max = max(route_dists)
                    if cur_max < best_max - 1e-12:
                        best_max = cur_max
                        best_routes = [route[:] for route in routes]
                        report_best_vrp(best_routes)
            if not improved:
                break
        # Shake: destroy and repair random subset
        for _ in range(shake_iterations):
            # Destroy: remove random subset of customers (10%-40%)
            num_remove = random.randint(max(1, (n-1)//10), max(1, (n-1)*4//10))
            customers = list(range(1, n))
            random.shuffle(customers)
            to_remove = customers[:num_remove]
            temp_routes = [route[:] for route in routes]
            temp_dists = route_dists[:]
            for c in to_remove:
                for r_idx in range(truck_count):
                    if c in temp_routes[r_idx]:
                        pos = temp_routes[r_idx].index(c)
                        pred = temp_routes[r_idx][pos-1]
                        succ = temp_routes[r_idx][pos+1]
                        temp_dists[r_idx] += distance_matrix[pred, succ] - distance_matrix[pred, c] - distance_matrix[c, succ]
                        temp_routes[r_idx].pop(pos)
                        break
            # Repair: using regret (deterministic)
            unassigned = to_remove[:]
            while unassigned:
                bests = []
                for c in unassigned:
                    best_new_max, best_route, best_pos, second_new_max = best_insertion(c, temp_routes, temp_dists)
                    if best_route == -1:
                        continue
                    regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
                    bests.append((-regret, c, best_route, best_pos, best_new_max))
                if not bests:
                    break
                bests.sort(key=lambda x: (x[0], x[1]))
                _, c, best_route, best_pos, _ = bests[0]
                temp_routes[best_route].insert(best_pos, c)
                temp_dists[best_route] = route_dist(temp_routes[best_route])
                unassigned.remove(c)
            # Apply intra 2-opt on all routes (only if some customers were removed)
            if to_remove:
                temp_routes, temp_dists = intra_2opt(temp_routes, temp_dists, list(range(truck_count)))
            # Accept new solution (even if worse)
            routes = [route[:] for route in temp_routes]
            route_dists = temp_dists[:]
            cur_max = max(route_dists)
            if cur_max < best_max - 1e-12:
                best_max = cur_max
                best_routes = [route[:] for route in routes]
                report_best_vrp(best_routes)
            # After shake, try immediate VND
            break  # only one shake per restart to keep loops bounded
    return best_routes