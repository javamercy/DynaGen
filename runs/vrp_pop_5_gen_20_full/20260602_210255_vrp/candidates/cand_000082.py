import numpy as np
from collections import defaultdict
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    
    def compute_route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d
    
    # Greedy construction: insert customers minimizing max route distance
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    unassigned.sort(key=lambda c: -distance_matrix[0][c])
    route_dists = [compute_route_dist(r) for r in routes]
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
                new_max = new_route_dist
                for other_idx, d in enumerate(route_dists):
                    if other_idx != r_idx and d > new_max:
                        new_max = d
                if new_max < best_new_max or (new_max == best_new_max and r_idx < best_route_idx):
                    best_new_max = new_max
                    best_route_idx = r_idx
                    best_pos = pos
        route = routes[best_route_idx]
        route.insert(best_pos, cust)
        route_dists[best_route_idx] = compute_route_dist(route)
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)
    
    # Helper functions for moves
    def apply_relocate(routes, r_idx, pos, cust, new_r_idx, new_pos):
        routes[r_idx].pop(pos)
        routes[new_r_idx].insert(new_pos, cust)
        route_dists[r_idx] = compute_route_dist(routes[r_idx])
        route_dists[new_r_idx] = compute_route_dist(routes[new_r_idx])
    
    def apply_swap(routes, r1, pos1, r2, pos2):
        cust1 = routes[r1][pos1]
        cust2 = routes[r2][pos2]
        routes[r1][pos1] = cust2
        routes[r2][pos2] = cust1
        route_dists[r1] = compute_route_dist(routes[r1])
        route_dists[r2] = compute_route_dist(routes[r2])
    
    def apply_two_opt(routes, r_idx, i, j):
        route = routes[r_idx]
        route[i:j+1] = reversed(route[i:j+1])
        route_dists[r_idx] = compute_route_dist(route)
    
    def apply_cross(routes, r1, pos1, r2, pos2):
        tail1 = routes[r1][pos1+1:]
        tail2 = routes[r2][pos2+1:]
        routes[r1] = routes[r1][:pos1+1] + tail2
        routes[r2] = routes[r2][:pos2+1] + tail1
        route_dists[r1] = compute_route_dist(routes[r1])
        route_dists[r2] = compute_route_dist(routes[r2])
    
    # Adaptive perturbation: move farthest customer from max route to random other route, repeat based on stagnation
    def strong_perturb(routes, route_dists, iteration):
        current_max = max(route_dists)
        max_routes = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
        # Number of moves: base 3 plus 1 per 100 iterations without improvement (but we don't track improvement here, just use iteration)
        num_moves = 3 + (iteration // 100)
        for _ in range(min(num_moves, 10)):  # cap
            valid_max = [i for i in max_routes if len(routes[i]) > 2]
            if not valid_max:
                break
            r_idx = random.choice(valid_max)
            route = routes[r_idx]
            customers = [node for node in route if node != 0]
            if not customers:
                break
            # farthest from depot
            cust = max(customers, key=lambda c: distance_matrix[0][c])
            pos = route.index(cust)
            route.pop(pos)
            route_dists[r_idx] = compute_route_dist(route)
            other_routes = [i for i in range(truck_count) if i != r_idx]
            if not other_routes:
                break
            new_r_idx = random.choice(other_routes)
            # Insert in random position
            insert_pos = random.randint(1, len(routes[new_r_idx]) - 1)
            routes[new_r_idx].insert(insert_pos, cust)
            route_dists[new_r_idx] = compute_route_dist(routes[new_r_idx])
            current_max = max(route_dists)
            max_routes = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
    
    max_restarts = 10
    for restart in range(max_restarts + 1):
        if restart > 0:
            routes = [list(r) for r in best_routes]
            route_dists = [compute_route_dist(r) for r in routes]
            strong_perturb(routes, route_dists, restart)
        else:
            routes = [list(r) for r in best_routes]
            route_dists = [compute_route_dist(r) for r in routes]
        
        max_iter = min(n * 20, 1000)
        no_improve_count = 0
        for iteration in range(max_iter):
            current_max = max(route_dists)
            max_route_idxs = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
            improved = False
            # 1. relocate from max route
            for r_idx in max_route_idxs:
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
                if max(route_dists) < best_max - 1e-12:
                    best_max = max(route_dists)
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                continue
            # 2. swap between max and another
            for r1 in max_route_idxs:
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
                if max(route_dists) < best_max - 1e-12:
                    best_max = max(route_dists)
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                continue
            # 3. 2-opt on max routes
            for r_idx in max_route_idxs:
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old_edges = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                        new_edges = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                        change = new_edges - old_edges
                        new_dist = route_dists[r_idx] + change
                        if new_dist < current_max - 1e-12:
                            route[i:j+1] = reversed(route[i:j+1])
                            route_dists[r_idx] = compute_route_dist(route)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                if max(route_dists) < best_max - 1e-12:
                    best_max = max(route_dists)
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                continue
            # 4. cross exchange between max and another
            for r1 in max_route_idxs:
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
                if max(route_dists) < best_max - 1e-12:
                    best_max = max(route_dists)
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                continue
            # No max-improving move found; accept a move that improves total distance (non-greedy escape)
            best_total_improve = float('inf')
            best_move = None
            # 1. relocate
            for r_idx in range(truck_count):
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
                            # total distance change
                            total_change = removal_change + insertion_change
                            if total_change < best_total_improve:
                                best_total_improve = total_change
                                best_move = ('relocate', r_idx, pos, cust, other_idx, insert_pos)
            # 2. swap
            for r1 in range(truck_count):
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
                            total_change = remove1_change + remove2_change + insert1_change + insert2_change
                            if total_change < best_total_improve:
                                best_total_improve = total_change
                                best_move = ('swap', r1, pos1, r2, pos2)
            # 3. 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old_edges = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                        new_edges = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                        change = new_edges - old_edges
                        if change < best_total_improve:
                            best_total_improve = change
                            best_move = ('2opt', r_idx, i, j)
            # 4. cross
            for r1 in range(truck_count):
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
                            total_change = delta1 + delta2
                            if total_change < best_total_improve:
                                best_total_improve = total_change
                                best_move = ('cross', r1, pos1, r2, pos2)
            if best_move is not None and best_total_improve < -1e-12:
                move_type = best_move[0]
                if move_type == 'relocate':
                    _, r_idx, pos, cust, new_r_idx, new_pos = best_move
                    apply_relocate(routes, r_idx, pos, cust, new_r_idx, new_pos)
                elif move_type == 'swap':
                    _, r1, pos1, r2, pos2 = best_move
                    apply_swap(routes, r1, pos1, r2, pos2)
                elif move_type == '2opt':
                    _, r_idx, i, j = best_move
                    route = routes[r_idx]
                    route[i:j+1] = reversed(route[i:j+1])
                    route_dists[r_idx] = compute_route_dist(route)
                elif move_type == 'cross':
                    _, r1, pos1, r2, pos2 = best_move
                    apply_cross(routes, r1, pos1, r2, pos2)
                improved = True
                if max(route_dists) < best_max - 1e-12:
                    best_max = max(route_dists)
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
            if not improved:
                break  # no improving move at all
        if max(route_dists) < best_max - 1e-12:
            best_max = max(route_dists)
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
    return best_routes