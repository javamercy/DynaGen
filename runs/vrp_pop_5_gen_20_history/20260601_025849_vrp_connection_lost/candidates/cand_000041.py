import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix

    def route_dist(route):
        d = 0.0
        for a, b in zip(route, route[1:]):
            d += dist[a][b]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    def construct_random():
        routes = [[0, 0] for _ in range(truck_count)]
        customers = list(range(1, n))
        random.shuffle(customers)
        for c in customers:
            r_idx = random.randrange(truck_count)
            pos = random.randint(1, len(routes[r_idx]) - 1)
            routes[r_idx].insert(pos, c)
        return routes

    def relocate_move(routes):
        src_idx = random.randrange(truck_count)
        if len(routes[src_idx]) <= 3:
            return None
        pos_src = random.randint(1, len(routes[src_idx]) - 2)
        cust = routes[src_idx][pos_src]
        tgt_idx = random.randrange(truck_count)
        pos_tgt = random.randint(1, len(routes[tgt_idx]) - 1)
        return ('relocate', src_idx, pos_src, cust, tgt_idx, pos_tgt)

    def swap_move(routes):
        idx1 = random.randrange(truck_count)
        if len(routes[idx1]) <= 3:
            return None
        pos1 = random.randint(1, len(routes[idx1]) - 2)
        idx2 = random.randrange(truck_count)
        if idx2 == idx1 or len(routes[idx2]) <= 3:
            return None
        pos2 = random.randint(1, len(routes[idx2]) - 2)
        return ('swap', idx1, pos1, idx2, pos2)

    def two_opt_move(routes):
        r_idx = random.randrange(truck_count)
        route = routes[r_idx]
        if len(route) <= 3:
            return None
        i = random.randint(1, len(route) - 3)
        j = random.randint(i + 1, len(route) - 2)
        return ('2opt', r_idx, i, j)

    def cross_exchange_move(routes):
        idx1 = random.randrange(truck_count)
        if len(routes[idx1]) < 4:
            return None
        idx2 = random.randrange(truck_count)
        if idx2 == idx1 or len(routes[idx2]) < 4:
            return None
        i1 = random.randint(1, len(routes[idx1]) - 3)
        j1 = random.randint(i1 + 1, len(routes[idx1]) - 2)
        i2 = random.randint(1, len(routes[idx2]) - 3)
        j2 = random.randint(i2 + 1, len(routes[idx2]) - 2)
        return ('cross', idx1, i1, j1, idx2, i2, j2)

    best_routes = construct_random()
    best_max = max_dist(best_routes)
    report_best_vrp(best_routes)
    current_routes = [list(r) for r in best_routes]
    current_max = best_max

    T = 100.0
    T_min = 1e-3
    alpha = 0.99
    max_iter = n * n * 10
    stall_limit = max(10, n // 5)
    it = 0
    stall = 0
    restart_count = 0
    max_restarts = 5

    while it < max_iter and T > T_min and restart_count < max_restarts:
        move_type = random.choice(['relocate', 'swap', '2opt', 'cross'])
        move = None
        if move_type == 'relocate':
            move = relocate_move(current_routes)
        elif move_type == 'swap':
            move = swap_move(current_routes)
        elif move_type == '2opt':
            move = two_opt_move(current_routes)
        elif move_type == 'cross':
            move = cross_exchange_move(current_routes)

        if move is None:
            it += 1
            continue

        candidate = [list(r) for r in current_routes]
        if move[0] == 'relocate':
            _, src_idx, pos_src, cust, tgt_idx, pos_tgt = move
            del candidate[src_idx][pos_src]
            candidate[tgt_idx].insert(pos_tgt, cust)
        elif move[0] == 'swap':
            _, idx1, pos1, idx2, pos2 = move
            cust1 = candidate[idx1][pos1]
            cust2 = candidate[idx2][pos2]
            candidate[idx1][pos1] = cust2
            candidate[idx2][pos2] = cust1
        elif move[0] == '2opt':
            _, r_idx, i, j = move
            candidate[r_idx][i:j + 1] = candidate[r_idx][i:j + 1][::-1]
        elif move[0] == 'cross':
            _, idx1, i1, j1, idx2, i2, j2 = move
            seg1 = candidate[idx1][i1:j1 + 1]
            seg2 = candidate[idx2][i2:j2 + 1]
            candidate[idx1][i1:j1 + 1] = seg2
            candidate[idx2][i2:j2 + 1] = seg1

        new_max = max_dist(candidate)
        delta = new_max - current_max
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_routes = candidate
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_routes = [list(r) for r in candidate]
                report_best_vrp(best_routes)
                stall = 0
            else:
                stall += 1
        else:
            stall += 1

        if stall >= stall_limit:
            current_routes = construct_random()
            current_max = max_dist(current_routes)
            stall = 0
            T = 100.0
            restart_count += 1

        T *= alpha
        it += 1

    final_routes = []
    for route in best_routes:
        if len(route) == 2:
            final_routes.append([0, 0])
        else:
            final_routes.append(route)
    return final_routes