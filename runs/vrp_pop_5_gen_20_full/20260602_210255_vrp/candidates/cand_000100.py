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

    # Greedy construction minimizing max route distance
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    unassigned.sort(key=lambda c: -distance_matrix[0][c])
    route_dists = [0.0 for _ in range(truck_count)]
    for cust in unassigned:
        best_new_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for r_idx in range(truck_count):
            route = routes[r_idx]
            for pos in range(1, len(route)):
                prev = route[pos-1]
                succ = route[pos]
                increase = distance_matrix[prev][cust] + distance_matrix[cust][succ] - distance_matrix[prev][succ]
                new_route_dist = route_dists[r_idx] + increase
                new_max = new_route_dist
                for other_idx, d in enumerate(route_dists):
                    if other_idx != r_idx and d > new_max:
                        new_max = d
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_route_idx = r_idx
                    best_pos = pos
        routes[best_route_idx].insert(best_pos, cust)
        route_dists[best_route_idx] = compute_route_dist(routes[best_route_idx])

    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)

    # helper functions
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

    # main search parameters (reduced to avoid timeout)
    max_restarts = min(5, n // 10)
    if max_restarts < 1:
        max_restarts = 1
    max_iter = min(n * 10, 1000)

    for restart in range(max_restarts + 1):
        if restart > 0:
            # perturb best solution
            routes = [list(r) for r in best_routes]
            route_dists = [compute_route_dist(r) for r in routes]
            # simple adaptive perturbation
            current_max = max(route_dists)
            current_min = min(route_dists)
            gap = current_max - current_min
            num_moves = min(10, max(1, int(gap / (best_max * 0.1)))) if best_max > 0 else 2
            for _ in range(num_moves):
                max_idxs = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12 and len(routes[i]) > 2]
                if not max_idxs:
                    break
                max_idx = random.choice(max_idxs)
                route = routes[max_idx]
                customers = [node for node in route if node != 0]
                if not customers:
                    break
                move_type = random.choice(['relocate', 'swap', 'two_opt'])
                if move_type == 'relocate':
                    cust = random.choice(customers)
                    pos = route.index(cust)
                    other_idx = random.choice([i for i in range(truck_count) if i != max_idx])
                    if other_idx is None:
                        continue
                    insert_pos = random.randint(1, len(routes[other_idx])-1)
                    apply_relocate(routes, max_idx, pos, cust, other_idx, insert_pos)
                elif move_type == 'swap':
                    other_idx = random.choice([i for i in range(truck_count) if i != max_idx and len(routes[i]) > 2])
                    if other_idx is None:
                        continue
                    pos1 = random.randint(1, len(route)-2)
                    pos2 = random.randint(1, len(routes[other_idx])-2)
                    apply_swap(routes, max_idx, pos1, other_idx, pos2)
                elif move_type == 'two_opt':
                    if len(route) > 3:
                        i = random.randint(1, len(route)-3)
                        j = random.randint(i+1, len(route)-2)
                        apply_two_opt(routes, max_idx, i, j)
                current_max = max(route_dists)
        else:
            routes = [list(r) for r in best_routes]
            route_dists = [compute_route_dist(r) for r in routes]

        stall = 0
        for iteration in range(max_iter):
            current_max = max(route_dists)
            max_route_idxs = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
            improved = False

            # 1. relocate from max routes with limited search
            if not improved:
                for r_idx in max_route_idxs:
                    route = routes[r_idx]
                    pos_list = list(range(1, len(route)-1))
                    random.shuffle(pos_list)
                    for pos in pos_list:
                        cust = route[pos]
                        prev = route[pos-1]
                        succ = route[pos+1]
                        removal_change = distance_matrix[prev][succ] - (distance_matrix[prev][cust] + distance_matrix[cust][succ])
                        new_dist_r = route_dists[r_idx] + removal_change
                        other_list = list(range(truck_count))
                        random.shuffle(other_list)
                        for other_idx in other_list:
                            if other_idx == r_idx:
                                continue
                            other_route = routes[other_idx]
                            ins_positions = list(range(1, len(other_route)))
                            random.shuffle(ins_positions)
                            for insert_pos in ins_positions:
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
                stall = 0
                if max(route_dists) < best_max - 1e-12:
                    best_max = max(route_dists)
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                continue

            # 2. swap between max and another route limited
            if not improved:
                for r1 in max_route_idxs:
                    route1 = routes[r1]
                    pos1_list = list(range(1, len(route1)-1))
                    random.shuffle(pos1_list)
                    for pos1 in pos1_list:
                        cust1 = route1[pos1]
                        for r2 in range(truck_count):
                            if r2 == r1:
                                continue
                            route2 = routes[r2]
                            pos2_list = list(range(1, len(route2)-1))
                            random.shuffle(pos2_list)
                            for pos2 in pos2_list:
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
                stall = 0
                if max(route_dists) < best_max - 1e-12:
                    best_max = max(route_dists)
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                continue

            # 3. 2-opt on max routes limited
            if not improved:
                for r_idx in max_route_idxs:
                    route = routes[r_idx]
                    i_list = list(range(1, len(route)-2))
                    random.shuffle(i_list)
                    for i in i_list:
                        j_list = list(range(i+1, len(route)-1))
                        random.shuffle(j_list)
                        for j in j_list:
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
                stall = 0
                if max(route_dists) < best_max - 1e-12:
                    best_max = max(route_dists)
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                continue

            # 4. No improvement, escape via random move
            stall += 1
            if random.random() < 0.3:
                move_type = random.choice(['relocate', 'swap', 'two_opt'])
                if move_type == 'relocate':
                    max_r_idx = random.choice(max_route_idxs)
                    route = routes[max_r_idx]
                    if len(route) > 2:
                        pos = random.randint(1, len(route)-2)
                        cust = route[pos]
                        other_idx = random.choice([i for i in range(truck_count) if i != max_r_idx])
                        if other_idx is not None:
                            insert_pos = random.randint(1, len(routes[other_idx])-1)
                            apply_relocate(routes, max_r_idx, pos, cust, other_idx, insert_pos)
                            improved = True
                elif move_type == 'swap':
                    max_r_idx = random.choice(max_route_idxs)
                    other_idx = random.choice([i for i in range(truck_count) if i != max_r_idx and len(routes[i]) > 2])
                    if other_idx is not None:
                        pos1 = random.randint(1, len(routes[max_r_idx])-2)
                        pos2 = random.randint(1, len(routes[other_idx])-2)
                        apply_swap(routes, max_r_idx, pos1, other_idx, pos2)
                        improved = True
                elif move_type == 'two_opt':
                    max_r_idx = random.choice(max_route_idxs)
                    route = routes[max_r_idx]
                    if len(route) > 3:
                        i = random.randint(1, len(route)-3)
                        j = random.randint(i+1, len(route)-2)
                        apply_two_opt(routes, max_r_idx, i, j)
                        improved = True
            if not improved:
                break

        if max(route_dists) < best_max - 1e-12:
            best_max = max(route_dists)
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)

    return best_routes