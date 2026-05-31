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

    def total_dist(routes):
        return sum(route_dist(r) for r in routes)

    def max_route_dist(routes):
        return max(route_dist(r) for r in routes)

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

    # Initial construction: regret heuristic
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
        _, c, best_route, best_pos, new_max = bests[0]
        route = routes[best_route]
        route.insert(best_pos, c)
        route_dists[best_route] = route_dist(route)
        unassigned.remove(c)

    best_routes = [route[:] for route in routes]
    best_max = max(route_dists)
    best_total = total_dist(routes)
    report_best_vrp(best_routes)

    # Tabu search parameters
    tabu_tenure_base = max(5, n // 5)
    max_iter = 100 * n
    no_improve_limit = 50 * n
    no_improve_count = 0
    tabu_list = {}  # key: (move_type, arg1, arg2, ...) encoded as tuple

    current_routes = [route[:] for route in routes]
    current_dists = route_dists[:]

    def evaluate_move(route_max, idx, new_max_dist, new_total_dist):
        other_max = 0.0
        for j, d in enumerate(current_dists):
            if j != idx[0] and j != idx[1] and d > other_max:
                other_max = d
        new_max = max(other_max, new_max_dist) if isinstance(new_max_dist, tuple) else max(other_max, new_max_dist)
        return new_max

    def generate_moves():
        moves = []
        max_dist = max(current_dists)
        max_idx = current_dists.index(max_dist)
        route_max = current_routes[max_idx]
        # relocate moves
        for i in range(1, len(route_max)-1):
            c = route_max[i]
            pred = route_max[i-1]
            succ = route_max[i+1]
            new_max_dist = current_dists[max_idx] - distance_matrix[pred, c] - distance_matrix[c, succ] + distance_matrix[pred, succ]
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = current_routes[other_idx]
                for pos in range(1, len(other_route)):
                    pred_o = other_route[pos-1]
                    succ_o = other_route[pos]
                    new_other = current_dists[other_idx] - distance_matrix[pred_o, succ_o] + distance_matrix[pred_o, c] + distance_matrix[c, succ_o]
                    other_max = 0.0
                    for j, d in enumerate(current_dists):
                        if j != max_idx and j != other_idx and d > other_max:
                            other_max = d
                    new_overall = max(other_max, new_max_dist, new_other)
                    new_total = best_total - current_dists[max_idx] - current_dists[other_idx] + new_max_dist + new_other
                    move = ('relocate', c, max_idx, i, other_idx, pos, new_max_dist, new_other)
                    key = ('relocate', c, max_idx, other_idx, i, pos)
                    moves.append((new_overall, new_total, move, key))
        # swap moves
        for i in range(1, len(route_max)-1):
            c1 = route_max[i]
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = current_routes[other_idx]
                for j in range(1, len(other_route)-1):
                    c2 = other_route[j]
                    old1 = current_dists[max_idx]
                    old2 = current_dists[other_idx]
                    pred1 = route_max[i-1]
                    succ1 = route_max[i+1]
                    new_dist_max = old1 - distance_matrix[pred1, c1] - distance_matrix[c1, succ1] + distance_matrix[pred1, c2] + distance_matrix[c2, succ1]
                    pred2 = other_route[j-1]
                    succ2 = other_route[j+1]
                    new_dist_other = old2 - distance_matrix[pred2, c2] - distance_matrix[c2, succ2] + distance_matrix[pred2, c1] + distance_matrix[c1, succ2]
                    other_max = 0.0
                    for k, d in enumerate(current_dists):
                        if k != max_idx and k != other_idx and d > other_max:
                            other_max = d
                    new_overall = max(other_max, new_dist_max, new_dist_other)
                    new_total = best_total - current_dists[max_idx] - current_dists[other_idx] + new_dist_max + new_dist_other
                    move = ('swap', c1, c2, max_idx, i, other_idx, j, new_dist_max, new_dist_other)
                    key = ('swap', max_idx, i, other_idx, j)
                    moves.append((new_overall, new_total, move, key))
        # 2-opt* moves (cross) - only if both routes end at depot
        for other_idx in range(truck_count):
            if other_idx == max_idx:
                continue
            other_route = current_routes[other_idx]
            for i in range(1, len(route_max)-1):
                for j in range(1, len(other_route)-1):
                    if route_max[-1] != 0 or other_route[-1] != 0:
                        continue
                    old1 = distance_matrix[route_max[i], route_max[i+1]]
                    old2 = distance_matrix[other_route[j], other_route[j+1]]
                    new1 = distance_matrix[route_max[i], other_route[j+1]]
                    new2 = distance_matrix[other_route[j], route_max[i+1]]
                    new_dist_max = current_dists[max_idx] - old1 + new1
                    new_dist_other = current_dists[other_idx] - old2 + new2
                    other_max = 0.0
                    for k, d in enumerate(current_dists):
                        if k != max_idx and k != other_idx and d > other_max:
                            other_max = d
                    new_overall = max(other_max, new_dist_max, new_dist_other)
                    new_total = best_total - current_dists[max_idx] - current_dists[other_idx] + new_dist_max + new_dist_other
                    move = ('cross', max_idx, i, other_idx, j, new_dist_max, new_dist_other)
                    key = ('cross', max_idx, i, other_idx, j)
                    moves.append((new_overall, new_total, move, key))
        return moves

    def apply_move(move):
        nonlocal current_routes, current_dists
        move_type = move[0]
        if move_type == 'relocate':
            _, c, max_idx, i, other_idx, pos, new_max_dist, new_other = move
            route_max = current_routes[max_idx]
            route_max.pop(i)
            current_routes[other_idx].insert(pos, c)
            current_dists[max_idx] = new_max_dist
            current_dists[other_idx] = new_other
        elif move_type == 'swap':
            _, c1, c2, max_idx, i, other_idx, j, new_dist_max, new_dist_other = move
            current_routes[max_idx][i] = c2
            current_routes[other_idx][j] = c1
            current_dists[max_idx] = new_dist_max
            current_dists[other_idx] = new_dist_other
        elif move_type == 'cross':
            _, max_idx, i, other_idx, j, new_dist_max, new_dist_other = move
            route_max = current_routes[max_idx]
            other_route = current_routes[other_idx]
            new_route_max = route_max[:i+1] + other_route[j+1:]
            new_route_other = other_route[:j+1] + route_max[i+1:]
            current_routes[max_idx] = new_route_max
            current_routes[other_idx] = new_route_other
            current_dists[max_idx] = new_dist_max
            current_dists[other_idx] = new_dist_other

    # Tabu search main loop
    for iteration in range(max_iter):
        moves = generate_moves()
        if not moves:
            break
        # Filter non-tabu moves and those that are tabu but satisfy aspiration
        non_tabu = []
        for new_overall, new_total, move, key in moves:
            if key not in tabu_list or tabu_list[key] <= iteration:
                non_tabu.append((new_overall, new_total, move, key))
            else:
                # Aspiration: if move leads to new global best, accept
                if new_overall < best_max or (new_overall == best_max and new_total < best_total):
                    non_tabu.append((new_overall, new_total, move, key))
        if not non_tabu:
            # All moves tabu, pick best overall anyway (diversification)
            non_tabu = moves
        # Select move using softmax over the best candidates (to add exploration)
        # Sort by (new_overall, new_total) and take top few
        non_tabu.sort(key=lambda x: (x[0], x[1]))
        candidates = non_tabu[:min(5, len(non_tabu))]
        if len(candidates) == 0:
            break
        # Softmax selection based on new_overall (lower is better)
        items = []
        for new_overall, new_total, move, key in candidates:
            items.append((new_overall, (new_overall, new_total, move, key)))
        # Use softmax with temperature decaying over iterations
        temperature = max(0.1, 10 * (1 - iteration / max_iter))
        best_overall = items[0][0]
        values = [best_overall - v for v, _ in items]
        max_val = max(values)
        shifted = [v - max_val for v in values]
        exp_vals = [np.exp(s) for s in shifted]
        total_exp = sum(exp_vals)
        if total_exp <= 1e-12:
            chosen = 0
        else:
            probs = [e / total_exp for e in exp_vals]
            r = random.random()
            cumulative = 0.0
            for i, prob in enumerate(probs):
                cumulative += prob
                if r <= cumulative:
                    chosen = i
                    break
            else:
                chosen = len(items)-1
        _, _, move, key = items[chosen][1]
        apply_move(move)
        # Update tabu list: add move and its reverse with tenure
        tenure = tabu_tenure_base + random.randint(0, 10)
        reverse_key = get_reverse_key(move)
        tabu_list[key] = iteration + tenure
        tabu_list[reverse_key] = iteration + tenure
        # Update best solution
        cur_max = max(current_dists)
        cur_total = total_dist(current_routes)
        if cur_max < best_max or (cur_max == best_max and cur_total < best_total):
            best_max = cur_max
            best_total = cur_total
            best_routes = [route[:] for route in current_routes]
            report_best_vrp(best_routes)
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= no_improve_limit:
            break
    return best_routes


def get_reverse_key(move):
    # For reverse tabu: e.g., relocate move reversed is also relocate with swapped roles
    move_type = move[0]
    if move_type == 'relocate':
        _, c, max_idx, i, other_idx, pos, _, _ = move
        # Reverse: moving c back from other_idx to max_idx at position i (needs original pos)
        # We can't fully reverse without storing more info; simplified: just store same key
        return ('relocate', c, other_idx, max_idx, pos, i)  # not exact but prevents cycles
    elif move_type == 'swap':
        _, c1, c2, max_idx, i, other_idx, j, _, _ = move
        return ('swap', other_idx, j, max_idx, i)  # swap back
    elif move_type == 'cross':
        _, max_idx, i, other_idx, j, _, _ = move
        return ('cross', other_idx, j, max_idx, i)  # reverse cross
    return move