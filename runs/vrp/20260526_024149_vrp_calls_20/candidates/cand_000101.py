import numpy as np
import random
from math import exp

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= len(customers):
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    best_routes = None
    best_max = float('inf')
    num_restarts = 5
    for restart in range(num_restarts):
        shuffled = customers[:]
        random.shuffle(shuffled)
        # Clarke-Wright savings construction
        routes = [[0, c, 0] for c in shuffled]
        while len(routes) > truck_count:
            best_saving = -1e9
            best_pair = None
            best_order = 0
            for i in range(len(routes)):
                for j in range(i+1, len(routes)):
                    ri = routes[i]
                    rj = routes[j]
                    if len(ri) <= 2 or len(rj) <= 2:
                        continue
                    last_i = ri[-2]
                    first_i = ri[1]
                    last_j = rj[-2]
                    first_j = rj[1]
                    s1 = distance_matrix[0][last_i] + distance_matrix[0][first_j] - distance_matrix[last_i][first_j]
                    s2 = distance_matrix[0][last_j] + distance_matrix[0][first_i] - distance_matrix[last_j][first_i]
                    if s1 > best_saving:
                        best_saving = s1
                        best_pair = (i, j)
                        best_order = 0
                    if s2 > best_saving:
                        best_saving = s2
                        best_pair = (i, j)
                        best_order = 1
            if best_pair is None:
                break
            i, j = best_pair
            if best_order == 0:
                new_route = routes[i][:-1] + routes[j][1:]
            else:
                new_route = routes[j][:-1] + routes[i][1:]
            if i < j:
                del routes[j]
                del routes[i]
            else:
                del routes[i]
                del routes[j]
            routes.append(new_route)
        while len(routes) < truck_count:
            routes.append([0, 0])

        current_routes = [list(r) for r in routes]
        current_dist = [route_distance(r, distance_matrix) for r in current_routes]
        current_max = max(current_dist)
        if current_max < best_max - 1e-9:
            best_max = current_max
            best_routes = [list(r) for r in current_routes]
            report_best_vrp(best_routes)

        # SA parameters with linear rank-based penalty
        T = 0.5 * current_max if current_max > 0 else 1.0
        alpha = 0.99
        max_iter = 2000
        for iteration in range(max_iter):
            # Find longest route
            max_idx = max(range(truck_count), key=lambda i: current_dist[i])
            move_type = random.randint(0, 3)
            improved = False
            if move_type == 0 and len(current_routes[max_idx]) > 3:
                r = current_routes[max_idx]
                best_imp = 0
                best_pair = None
                for i in range(1, len(r)-2):
                    for j in range(i+1, len(r)-1):
                        if j - i == 1:
                            continue
                        new_route = r[:i] + r[i:j+1][::-1] + r[j+1:]
                        new_dist = route_distance(new_route, distance_matrix)
                        old_dist = route_distance(r, distance_matrix)
                        if new_dist < old_dist - 1e-9:
                            improvement = old_dist - new_dist
                            if improvement > best_imp:
                                best_imp = improvement
                                best_pair = (i, j, new_route, new_dist)
                if best_pair:
                    i, j, new_route, new_dist = best_pair
                    new_dists = current_dist.copy()
                    new_dists[max_idx] = new_dist
                    new_max = max(new_dists)
                    delta = new_max - current_max
                    if delta < 0 or random.random() < exp(-delta / T):
                        current_routes[max_idx] = new_route
                        current_dist[max_idx] = new_dist
                        current_max = new_max
                        if current_max < best_max - 1e-9:
                            best_max = current_max
                            best_routes = [list(r) for r in current_routes]
                            report_best_vrp(best_routes)
                        improved = True
            elif move_type == 1 and len(current_routes[max_idx]) > 2:
                r_max = current_routes[max_idx]
                best_delta = float('inf')
                best_move = None
                for pos in range(1, len(r_max)-1):
                    cust = r_max[pos]
                    new_max_route = r_max[:pos] + r_max[pos+1:]
                    new_max_dist = route_distance(new_max_route, distance_matrix)
                    for other_idx in range(truck_count):
                        if other_idx == max_idx:
                            continue
                        other_route = current_routes[other_idx]
                        for insert_pos in range(1, len(other_route)):
                            new_other_route = other_route[:insert_pos] + [cust] + other_route[insert_pos:]
                            new_other_dist = route_distance(new_other_route, distance_matrix)
                            new_dists = current_dist.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max = max(new_dists)
                            delta = new_max - current_max
                            if delta < best_delta:
                                best_delta = delta
                                best_move = (max_idx, new_max_route, other_idx, new_other_route, new_max, new_dists)
                if best_move is not None and (best_delta < 0 or random.random() < exp(-best_delta / T)):
                    max_idx, new_max_route, other_idx, new_other_route, new_max, new_dists = best_move
                    current_routes[max_idx] = new_max_route
                    current_routes[other_idx] = new_other_route
                    current_dist = new_dists
                    current_max = new_max
                    if current_max < best_max - 1e-9:
                        best_max = current_max
                        best_routes = [list(r) for r in current_routes]
                        report_best_vrp(best_routes)
                    improved = True
            elif move_type == 2 and len(current_routes[max_idx]) > 2:
                r_max = current_routes[max_idx]
                best_delta = float('inf')
                best_move = None
                for other_idx in range(truck_count):
                    if other_idx == max_idx or len(current_routes[other_idx]) <= 2:
                        continue
                    other_route = current_routes[other_idx]
                    for pos_max in range(1, len(r_max)-1):
                        cust_a = r_max[pos_max]
                        for pos_other in range(1, len(other_route)-1):
                            cust_b = other_route[pos_other]
                            new_max_route = r_max.copy()
                            new_max_route[pos_max] = cust_b
                            new_max_dist = route_distance(new_max_route, distance_matrix)
                            new_other_route = other_route.copy()
                            new_other_route[pos_other] = cust_a
                            new_other_dist = route_distance(new_other_route, distance_matrix)
                            new_dists = current_dist.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max = max(new_dists)
                            delta = new_max - current_max
                            if delta < best_delta:
                                best_delta = delta
                                best_move = (max_idx, new_max_route, other_idx, new_other_route, new_max, new_dists)
                if best_move is not None and (best_delta < 0 or random.random() < exp(-best_delta / T)):
                    max_idx, new_max_route, other_idx, new_other_route, new_max, new_dists = best_move
                    current_routes[max_idx] = new_max_route
                    current_routes[other_idx] = new_other_route
                    current_dist = new_dists
                    current_max = new_max
                    if current_max < best_max - 1e-9:
                        best_max = current_max
                        best_routes = [list(r) for r in current_routes]
                        report_best_vrp(best_routes)
                    improved = True
            elif move_type == 3 and len(current_routes[max_idx]) > 2:
                r_max = current_routes[max_idx]
                best_delta = float('inf')
                best_move = None
                for other_idx in range(truck_count):
                    if other_idx == max_idx or len(current_routes[other_idx]) <= 2:
                        continue
                    other_route = current_routes[other_idx]
                    for i in range(1, len(r_max)-2):
                        for j in range(1, len(other_route)-2):
                            new_r_max = r_max[:i+1] + other_route[j+1:-1] + [0]
                            new_other = other_route[:j+1] + r_max[i+1:-1] + [0]
                            new_max_dist = route_distance(new_r_max, distance_matrix)
                            new_other_dist = route_distance(new_other, distance_matrix)
                            new_dists = current_dist.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max = max(new_dists)
                            delta = new_max - current_max
                            if delta < best_delta:
                                best_delta = delta
                                best_move = (max_idx, new_r_max, other_idx, new_other, new_max, new_dists)
                if best_move is not None and (best_delta < 0 or random.random() < exp(-best_delta / T)):
                    max_idx, new_r_max, other_idx, new_other, new_max, new_dists = best_move
                    current_routes[max_idx] = new_r_max
                    current_routes[other_idx] = new_other
                    current_dist = new_dists
                    current_max = new_max
                    if current_max < best_max - 1e-9:
                        best_max = current_max
                        best_routes = [list(r) for r in current_routes]
                        report_best_vrp(best_routes)
                    improved = True
            if not improved:
                # Ejection chain perturbation with linear rank-based adaptive penalty
                r_max = current_routes[max_idx]
                if len(r_max) <= 3:
                    continue
                contributions = []
                for k in range(1, len(r_max)-1):
                    prev = r_max[k-1]
                    curr = r_max[k]
                    nxt = r_max[k+1]
                    contrib = distance_matrix[prev][curr] + distance_matrix[curr][nxt] - distance_matrix[prev][nxt]
                    contributions.append((contrib, r_max[k]))
                contributions.sort(reverse=True)
                num_eject = random.randint(1, min(3, len(r_max)-2))
                ejected = [c[1] for c in contributions[:num_eject]]
                new_route = [x for x in r_max if x not in ejected]
                # Linear rank-based adaptive penalty
                dist_sorted = sorted(enumerate(current_dist), key=lambda x: x[1])
                rank = {idx: i for i, (idx, _) in enumerate(dist_sorted)}  # 0 for shortest
                n_trucks = truck_count
                for cust in ejected:
                    best_effective = float('inf')
                    best_route_idx = -1
                    best_pos = -1
                    for other_idx in range(truck_count):
                        if other_idx == max_idx:
                            continue
                        other_route = current_routes[other_idx]
                        for pos in range(1, len(other_route)):
                            new_other_route = other_route[:pos] + [cust] + other_route[pos:]
                            new_dist = route_distance(new_other_route, distance_matrix)
                            old_dist = route_distance(other_route, distance_matrix)
                            increase = new_dist - old_dist
                            # Linear penalty: 1 + 0.5 * (rank / (n_trucks-1))
                            penalty = 1.0 + 0.5 * (rank[other_idx] / (n_trucks - 1)) if n_trucks > 1 else 1.0
                            effective_increase = increase * penalty
                            if effective_increase < best_effective:
                                best_effective = effective_increase
                                best_route_idx = other_idx
                                best_pos = pos
                    if best_route_idx != -1:
                        current_routes[best_route_idx] = current_routes[best_route_idx][:best_pos] + [cust] + current_routes[best_route_idx][best_pos:]
                current_routes[max_idx] = new_route
                current_dist = [route_distance(r, distance_matrix) for r in current_routes]
                current_max = max(current_dist)
            T *= alpha
            if T < 1e-6:
                break
    report_best_vrp(best_routes if best_routes is not None else current_routes)
    return best_routes if best_routes is not None else current_routes