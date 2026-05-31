import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    dist = distance_matrix

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d

    # Initial construction: greedy insertion minimizing max distance (same as parent)
    customers = list(range(1, n))
    random.shuffle(customers)
    routes = [[0, 0] for _ in range(truck_count)]
    for cust in customers:
        best_truck = None
        best_pos = None
        best_max = float('inf')
        best_total = float('inf')
        for t, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_routes = routes[:t] + [new_route] + routes[t+1:]
                new_max = max(route_distance(r) for r in new_routes)
                new_total = sum(route_distance(r) for r in new_routes)
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max = new_max
                    best_total = new_total
                    best_truck = t
                    best_pos = pos
        routes[best_truck].insert(best_pos, cust)

    current_routes = [list(r) for r in routes]
    best_routes = [list(r) for r in routes]
    best_max = max(route_distance(r) for r in best_routes)
    report_best_vrp(best_routes)

    # Tabu search parameters
    max_iter = max(100, 5 * n)
    tabu_tenure = 10
    tabu = dict()  # customer -> iteration until which it is tabu
    iteration = 0
    no_improve = 0
    restart_threshold = int(0.1 * max_iter)

    # Helper to evaluate a set of routes
    def evaluate(routes):
        max_dist = -1.0
        total = 0.0
        for r in routes:
            d = route_distance(r)
            if d > max_dist:
                max_dist = d
            total += d
        return max_dist, total

    def get_move_effects(routes, move):
        # move: tuple (type, cust1, cust2, route1, route2, pos1, pos2) ... we'll process differently
        pass

    # Instead, generate all moves in each iteration
    while iteration < max_iter:
        # Generate all possible moves
        best_move = None
        best_new_max = None
        best_new_total = None

        # 1. Intra-route 2-opt moves
        for t, route in enumerate(current_routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_routes = current_routes[:t] + [new_route] + current_routes[t+1:]
                    new_max, new_total = evaluate(new_routes)
                    # No tabu for 2-opt
                    if best_move is None or new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                        best_move = ('2opt', t, i, j)
                        best_new_max = new_max
                        best_new_total = new_total
                    elif new_max == best_new_max and new_total == best_new_total:
                        # deterministic tie-break: prefer lower route index, then lower i, then lower j
                        pass

        # 2. Inter-route relocate moves
        for cust in range(1, n):
            # find current route and position of cust
            src_t = -1
            src_pos = -1
            for t, route in enumerate(current_routes):
                for p, c in enumerate(route):
                    if c == cust:
                        src_t = t
                        src_pos = p
                        break
                if src_t != -1:
                    break
            if src_t == -1:
                continue
            for dst_t, dst_route in enumerate(current_routes):
                if dst_t == src_t:
                    continue
                for dst_pos in range(1, len(dst_route)):  # can insert before any internal node including before depot at end? Actually depot at both ends, so positions 1..len(route)-1
                    # Remove cust from src and insert at dst_pos in dst
                    new_src_route = [r for r in current_routes[src_t] if r != cust]  # this removes all occurrences, but cust appears once
                    # Actually remove exactly one occurrence:
                    new_src_route = current_routes[src_t][:src_pos] + current_routes[src_t][src_pos+1:]
                    new_dst_route = dst_route[:dst_pos] + [cust] + dst_route[dst_pos:]
                    new_routes = current_routes[:]
                    new_routes[src_t] = new_src_route
                    new_routes[dst_t] = new_dst_route
                    new_max, new_total = evaluate(new_routes)
                    # Check tabu: cust is tabu if iteration < tabu.get(cust, -1)
                    tabu_cust = tabu.get(cust, -1)
                    is_tabu = iteration < tabu_cust
                    if is_tabu:
                        # Aspiration: accept if new_max < best_max
                        if new_max < best_max:
                            is_tabu = False
                    if not is_tabu:
                        if best_move is None or new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                            best_move = ('relocate', cust, src_t, src_pos, dst_t, dst_pos)
                            best_new_max = new_max
                            best_new_total = new_total

        # 3. Inter-route swap moves
        for cust1 in range(1, n):
            for cust2 in range(cust1+1, n):
                # find routes and positions
                t1 = p1 = t2 = p2 = -1
                for t, route in enumerate(current_routes):
                    for p, c in enumerate(route):
                        if c == cust1:
                            t1 = t; p1 = p
                        if c == cust2:
                            t2 = t; p2 = p
                if t1 == -1 or t2 == -1:
                    continue
                if t1 == t2:
                    continue
                # swap
                new_routes = [list(r) for r in current_routes]
                new_routes[t1][p1] = cust2
                new_routes[t2][p2] = cust1
                new_max, new_total = evaluate(new_routes)
                # Check tabu for both customers
                tabu1 = iteration < tabu.get(cust1, -1)
                tabu2 = iteration < tabu.get(cust2, -1)
                is_tabu = tabu1 or tabu2
                if is_tabu:
                    if new_max < best_max:
                        is_tabu = False
                if not is_tabu:
                    if best_move is None or new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                        best_move = ('swap', cust1, cust2, t1, p1, t2, p2)
                        best_new_max = new_max
                        best_new_total = new_total

        if best_move is None:
            break

        # Apply best move
        mtype = best_move[0]
        if mtype == '2opt':
            t, i, j = best_move[1:4]
            route = current_routes[t]
            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
            current_routes[t] = new_route
        elif mtype == 'relocate':
            cust, src_t, src_pos, dst_t, dst_pos = best_move[1:6]
            # remove from src
            current_routes[src_t].pop(src_pos)
            # insert into dst
            current_routes[dst_t].insert(dst_pos, cust)
            # update tabu
            tabu[cust] = iteration + tabu_tenure
        elif mtype == 'swap':
            cust1, cust2, t1, p1, t2, p2 = best_move[1:7]
            current_routes[t1][p1] = cust2
            current_routes[t2][p2] = cust1
            tabu[cust1] = iteration + tabu_tenure
            tabu[cust2] = iteration + tabu_tenure

        # Update evaluation
        current_max = best_new_max
        current_total = best_new_total

        # Check if new best
        if current_max < best_max - 1e-9 or (abs(current_max - best_max) < 1e-9 and current_total < sum(route_distance(r) for r in best_routes)):
            best_max = current_max
            best_routes = [list(r) for r in current_routes]
            report_best_vrp(best_routes)
            no_improve = 0
        else:
            no_improve += 1

        iteration += 1

        # Restart if stuck
        if no_improve >= restart_threshold:
            # Random perturbation: relocate random customers to random positions
            num_perturb = max(1, int(0.2 * (n-1)))
            for _ in range(num_perturb):
                cust = random.randint(1, n-1)
                # find its current route and position
                src_t = src_p = -1
                for t, route in enumerate(current_routes):
                    for p, c in enumerate(route):
                        if c == cust:
                            src_t = t
                            src_p = p
                            break
                    if src_t != -1:
                        break
                if src_t == -1:
                    continue
                # remove from src
                current_routes[src_t].pop(src_p)
                # insert at random position in random different route
                dst_t = random.choice([t for t in range(truck_count) if t != src_t])
                dst_pos = random.randint(1, len(current_routes[dst_t])-1)
                current_routes[dst_t].insert(dst_pos, cust)
            # Reset tabu and iteration limit
            tabu.clear()
            no_improve = 0
            # Evaluate new current
            current_max = max(route_distance(r) for r in current_routes)
            current_total = sum(route_distance(r) for r in current_routes)
            if current_max < best_max or (current_max == best_max and current_total < sum(route_distance(r) for r in best_routes)):
                best_max = current_max
                best_routes = [list(r) for r in current_routes]
                report_best_vrp(best_routes)

    return best_routes