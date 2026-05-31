import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    max_dist = np.max(distance_matrix)

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def balance_routes(routes, lengths):
        improved = True
        max_balance_iter = n
        it = 0
        while improved and it < max_balance_iter:
            improved = False
            it += 1
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if max_idx == min_idx or lengths[max_idx] == lengths[min_idx]:
                break
            max_route = routes[max_idx]
            best_cust = None
            best_overall_reduction = 0
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max_route = max_route[:pos] + max_route[pos+1:]
                new_max_len = route_distance(new_max_route)
                min_route = routes[min_idx]
                best_insertion_len = float('inf')
                best_pos = -1
                for p in range(1, len(min_route)):
                    new_min_route = min_route[:p] + [cust] + min_route[p:]
                    l = route_distance(new_min_route)
                    if l < best_insertion_len:
                        best_insertion_len = l
                        best_pos = p
                new_min_route = min_route[:best_pos] + [cust] + min_route[best_pos:]
                new_min_len = route_distance(new_min_route)
                other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                new_max_global = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
                old_max_global = max(lengths)
                reduction = old_max_global - new_max_global
                if reduction > best_overall_reduction:
                    best_overall_reduction = reduction
                    best_cust = (cust, best_pos)
            if best_cust is not None and best_overall_reduction > 0:
                cust, best_insert_pos = best_cust
                new_max = [node for node in max_route if node != cust]
                min_route = routes[min_idx]
                new_min = min_route[:best_insert_pos] + [cust] + min_route[best_insert_pos:]
                routes[max_idx] = new_max
                routes[min_idx] = new_min
                lengths[max_idx] = route_distance(new_max)
                lengths[min_idx] = route_distance(new_min)
                improved = True
            else:
                break
        return routes, lengths

    def regret_insertion_construction(k=3):
        routes = [[0, 0] for _ in range(truck_count)]
        unvisited = set(customers)
        while unvisited:
            best_cust = None
            best_regret = -float('inf')
            best_inc = float('inf')
            best_route_idx = -1
            best_pos = -1
            for cust in unvisited:
                incs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        incs.append((inc, pos, r_idx))
                incs.sort(key=lambda x: x[0])
                if len(incs) >= k:
                    regret = sum(incs[i][0] - incs[0][0] for i in range(1, k))
                else:
                    regret = 0.0
                inc = incs[0][0]
                pos = incs[0][1]
                r_idx = incs[0][2]
                if regret > best_regret or (regret == best_regret and inc < best_inc):
                    best_regret = regret
                    best_inc = inc
                    best_cust = cust
                    best_route_idx = r_idx
                    best_pos = pos
            routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        return routes

    def two_opt(route):
        improved = True
        max_iter = 5
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        route = new_route
                        improved = True
        return route

    def double_bridge(route):
        if len(route) < 7:
            return route
        a = random.randint(1, len(route)-6)
        b = random.randint(a+1, len(route)-5)
        c = random.randint(b+1, len(route)-4)
        d = random.randint(c+1, len(route)-3)
        new_route = route[:a] + route[c:d] + route[b:c] + route[a:b] + route[d:]
        return new_route

    # Build initial
    routes = regret_insertion_construction(k=3)
    lengths = [route_distance(r) for r in routes]
    routes, lengths = balance_routes(routes, lengths)
    current_max = max(lengths)
    current_total = sum(lengths)
    best_routes = [r[:] for r in routes]
    best_max = current_max
    best_total = current_total
    report_best_vrp(best_routes)

    # Simulated Annealing parameters
    initial_temp = 0.1 * current_max
    temp = initial_temp
    cooling_rate = 0.99
    max_iter = n * truck_count * 3
    stagnation = 0
    # Main loop
    for iteration in range(max_iter):
        # Choose move type
        move_type = random.choice(['relocate', 'swap', '2opt'])
        # Generate a random move
        if move_type == 'relocate':
            # Pick random customer
            cust = random.choice(customers)
            # Find its current route
            src_idx = None
            src_pos = None
            for r_idx, route in enumerate(routes):
                if cust in route:
                    src_idx = r_idx
                    src_pos = route.index(cust)
                    break
            if src_idx is None:
                continue
            # Remove customer from source
            new_src = routes[src_idx][:src_pos] + routes[src_idx][src_pos+1:]
            # Pick destination route (different from source)
            dst_idx = random.choice([i for i in range(truck_count) if i != src_idx and len(routes[i]) > 2])
            if dst_idx is None:
                continue
            # Pick insertion position
            ins_pos = random.randint(1, len(routes[dst_idx])-1)
            new_dst = routes[dst_idx][:ins_pos] + [cust] + routes[dst_idx][ins_pos:]
            new_routes = [r[:] for r in routes]
            new_routes[src_idx] = new_src
            new_routes[dst_idx] = new_dst
        elif move_type == 'swap':
            # Pick two routes
            i, j = random.sample(range(truck_count), 2)
            if len(routes[i]) <= 2 or len(routes[j]) <= 2:
                continue
            # Pick positions (not depot)
            ipos = random.randint(1, len(routes[i])-2)
            jpos = random.randint(1, len(routes[j])-2)
            cust_i = routes[i][ipos]
            cust_j = routes[j][jpos]
            new_i = routes[i][:ipos] + [cust_j] + routes[i][ipos+1:]
            new_j = routes[j][:jpos] + [cust_i] + routes[j][jpos+1:]
            new_routes = [r[:] for r in routes]
            new_routes[i] = new_i
            new_routes[j] = new_j
        else:  # 2opt
            # Pick a random route
            r_idx = random.randrange(truck_count)
            route = routes[r_idx]
            if len(route) < 4:
                continue
            i = random.randint(1, len(route)-3)
            j = random.randint(i+1, len(route)-2)
            new_seg = route[i:j+1][::-1]
            new_route = route[:i] + new_seg + route[j+1:]
            new_routes = [r[:] for r in routes]
            new_routes[r_idx] = new_route
        # Evaluate new solution
        new_lengths = [route_distance(r) for r in new_routes]
        new_max = max(new_lengths)
        new_total = sum(new_lengths)
        # Acceptance
        delta = new_max - current_max
        if delta < 0 or (delta == 0 and new_total < current_total) or random.random() < math.exp(-delta / temp):
            routes = new_routes
            lengths = new_lengths
            current_max = new_max
            current_total = new_total
            if current_max < best_max or (current_max == best_max and current_total < best_total):
                best_max = current_max
                best_total = current_total
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
            stagnation = 0
        else:
            stagnation += 1
        # Update temperature
        temp *= cooling_rate
        # Reheat if stagnant
        if stagnation > n:
            temp = min(initial_temp, temp * 2)
            stagnation = 0
            # Apply double bridge to a random long route to diversify
            long_routes = [i for i, route in enumerate(routes) if len(route) >= 7]
            if long_routes:
                idx = random.choice(long_routes)
                new_route = double_bridge(routes[idx])
                new_routes = [r[:] for r in routes]
                new_routes[idx] = new_route
                # Apply local search 2-opt on that route
                new_route = two_opt(new_route)
                new_routes[idx] = new_route
                new_lengths = [route_distance(r) for r in new_routes]
                new_max = max(new_lengths)
                new_total = sum(new_lengths)
                if new_max < current_max or (new_max == current_max and new_total < current_total):
                    routes = new_routes
                    lengths = new_lengths
                    current_max = new_max
                    current_total = new_total
                    if current_max < best_max or (current_max == best_max and current_total < best_total):
                        best_max = current_max
                        best_total = current_total
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
    return best_routes