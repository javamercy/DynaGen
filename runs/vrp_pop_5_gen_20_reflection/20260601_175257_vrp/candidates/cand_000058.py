import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    best_routes = None
    best_max = float('inf')

    def compute_route_length(route):
        if len(route) < 2:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    # Initial construction
    shuffled = customers[:]
    random.shuffle(shuffled)
    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0 for _ in range(truck_count)]
    unassigned = set(shuffled)

    # Regret-3 insertion minimizing max route distance
    while unassigned:
        best_customer = None
        best_route_idx = -1
        best_pos = -1
        best_regret = -1.0
        best_new_max = float('inf')
        best_route_length = float('inf')

        for c in unassigned:
            insertions = []
            for r_idx, route in enumerate(routes):
                best_local_cost = float('inf')
                best_local_pos = -1
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    increase = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                    new_len = route_lengths[r_idx] + increase
                    new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                    if new_max < best_local_cost or (new_max == best_local_cost and route_lengths[r_idx] < (best_route_length if best_route_idx != -1 else float('inf'))):
                        best_local_cost = new_max
                        best_local_pos = pos
                insertions.append((best_local_cost, best_local_pos, r_idx))
            insertions.sort(key=lambda x: (x[0], route_lengths[x[2]], x[1]))
            if not insertions:
                continue
            best_cost = insertions[0][0]
            second_best_cost = insertions[1][0] if len(insertions) > 1 else best_cost
            third_best_cost = insertions[2][0] if len(insertions) > 2 else best_cost
            regret = second_best_cost - best_cost
            if len(insertions) > 2:
                regret += (third_best_cost - best_cost) / 2.0
            if regret > best_regret or (regret == best_regret and (best_cost < best_new_max or (best_cost == best_new_max and route_lengths[insertions[0][2]] < best_route_length))):
                best_regret = regret
                best_customer = c
                best_new_max = best_cost
                best_route_idx = insertions[0][2]
                best_pos = insertions[0][1]
                best_route_length = route_lengths[best_route_idx]

        if best_customer is None:
            best_customer = next(iter(unassigned))
            best_new_max = float('inf')
            best_route_idx = -1
            best_pos = -1
            best_route_length = float('inf')
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    increase = distance_matrix[prev, best_customer] + distance_matrix[best_customer, nxt] - distance_matrix[prev, nxt]
                    new_len = route_lengths[r_idx] + increase
                    new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                    if new_max < best_new_max or (new_max == best_new_max and route_lengths[r_idx] < best_route_length):
                        best_new_max = new_max
                        best_route_idx = r_idx
                        best_pos = pos
                        best_route_length = route_lengths[r_idx]

        route = routes[best_route_idx]
        route.insert(best_pos, best_customer)
        route_lengths[best_route_idx] = compute_route_length(route)
        unassigned.remove(best_customer)
        report_best_vrp(routes)

    # Local search (minimax acceptance)
    def local_search(routes, route_lengths):
        max_iterations = 5 * n * truck_count + 10
        improved = True
        while improved and max_iterations > 0:
            improved = False
            max_iterations -= 1
            # 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                        new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        if new < old - 1e-9:
                            old_len = route_lengths[r_idx]
                            new_len = old_len - (old - new)
                            old_max = max(route_lengths)
                            new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                            if new_max < old_max - 1e-9:
                                route[i:j+1] = reversed(route[i:j+1])
                                route_lengths[r_idx] = new_len
                                improved = True
                                report_best_vrp(routes)
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # relocate
            for r_from in range(truck_count):
                route_from = routes[r_from]
                if len(route_from) <= 2:
                    continue
                for idx_c in range(1, len(route_from)-1):
                    c = route_from[idx_c]
                    prev = route_from[idx_c-1]
                    nxt = route_from[idx_c+1]
                    cost_remove = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                    new_len_from = route_lengths[r_from] - cost_remove
                    for r_to in range(truck_count):
                        if r_to == r_from:
                            continue
                        route_to = routes[r_to]
                        for pos in range(1, len(route_to)):
                            prev_to = route_to[pos-1]
                            nxt_to = route_to[pos]
                            cost_insert = distance_matrix[prev_to, c] + distance_matrix[c, nxt_to] - distance_matrix[prev_to, nxt_to]
                            new_len_to = route_lengths[r_to] + cost_insert
                            old_max = max(route_lengths)
                            new_max = max(route_lengths[:r_from] + [new_len_from] + route_lengths[r_from+1:r_to] + [new_len_to] + route_lengths[r_to+1:])
                            if new_max < old_max - 1e-9:
                                route_from.pop(idx_c)
                                route_lengths[r_from] = new_len_from
                                route_to.insert(pos, c)
                                route_lengths[r_to] = new_len_to
                                improved = True
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # swap
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 2:
                    continue
                for idx1 in range(1, len(route1)-1):
                    c1 = route1[idx1]
                    prev1 = route1[idx1-1]
                    nxt1 = route1[idx1+1]
                    cost_remove1 = distance_matrix[prev1, c1] + distance_matrix[c1, nxt1] - distance_matrix[prev1, nxt1]
                    for r2 in range(r1+1, truck_count):
                        route2 = routes[r2]
                        if len(route2) <= 2:
                            continue
                        for idx2 in range(1, len(route2)-1):
                            c2 = route2[idx2]
                            prev2 = route2[idx2-1]
                            nxt2 = route2[idx2+1]
                            cost_remove2 = distance_matrix[prev2, c2] + distance_matrix[c2, nxt2] - distance_matrix[prev2, nxt2]
                            cost_insert1 = distance_matrix[prev1, c2] + distance_matrix[c2, nxt1] - distance_matrix[prev1, nxt1]
                            new_len1 = route_lengths[r1] - cost_remove1 + cost_insert1
                            cost_insert2 = distance_matrix[prev2, c1] + distance_matrix[c1, nxt2] - distance_matrix[prev2, nxt2]
                            new_len2 = route_lengths[r2] - cost_remove2 + cost_insert2
                            old_max = max(route_lengths)
                            new_max = max(route_lengths[:r1] + [new_len1] + route_lengths[r1+1:r2] + [new_len2] + route_lengths[r2+1:])
                            if new_max < old_max - 1e-9:
                                del route1[idx1]
                                del route2[idx2]
                                route1.insert(idx1, c2)
                                route2.insert(idx2, c1)
                                route_lengths[r1] = new_len1
                                route_lengths[r2] = new_len2
                                improved = True
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    local_search(routes, route_lengths)
    current_max = max(route_lengths)
    if current_max < best_max:
        best_max = current_max
        best_routes = [route[:] for route in routes]

    # VNS: varying removal intensity
    removal_intensities = [0.1, 0.15, 0.2, 0.25, 0.3]
    max_vns_cycles = 5
    for intensity in removal_intensities:
        for _ in range(max_vns_cycles):
            # Shake: remove random customers
            num_to_remove = max(1, int(intensity * (n - 1)))
            all_customers = [cust for route in routes for cust in route[1:-1]]
            if len(all_customers) < num_to_remove:
                num_to_remove = len(all_customers)
            if num_to_remove == 0:
                continue
            removed = random.sample(all_customers, num_to_remove)
            for c in removed:
                for r_idx, route in enumerate(routes):
                    if c in route:
                        idx = route.index(c)
                        prev = route[idx-1]
                        nxt = route[idx+1]
                        cost_remove = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                        route.pop(idx)
                        route_lengths[r_idx] -= cost_remove
                        break
            unassigned = set(removed)
            # Reinsert with same regret-3 heuristic
            while unassigned:
                best_customer = None
                best_route_idx = -1
                best_pos = -1
                best_regret = -1.0
                best_new_max = float('inf')
                best_route_length = float('inf')

                for c in unassigned:
                    insertions = []
                    for r_idx, route in enumerate(routes):
                        best_local_cost = float('inf')
                        best_local_pos = -1
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            increase = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                            new_len = route_lengths[r_idx] + increase
                            new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                            if new_max < best_local_cost or (new_max == best_local_cost and route_lengths[r_idx] < (best_route_length if best_route_idx != -1 else float('inf'))):
                                best_local_cost = new_max
                                best_local_pos = pos
                        insertions.append((best_local_cost, best_local_pos, r_idx))
                    insertions.sort(key=lambda x: (x[0], route_lengths[x[2]], x[1]))
                    if not insertions:
                        continue
                    best_cost = insertions[0][0]
                    second_best_cost = insertions[1][0] if len(insertions) > 1 else best_cost
                    third_best_cost = insertions[2][0] if len(insertions) > 2 else best_cost
                    regret = second_best_cost - best_cost
                    if len(insertions) > 2:
                        regret += (third_best_cost - best_cost) / 2.0
                    if regret > best_regret or (regret == best_regret and (best_cost < best_new_max or (best_cost == best_new_max and route_lengths[insertions[0][2]] < best_route_length))):
                        best_regret = regret
                        best_customer = c
                        best_new_max = best_cost
                        best_route_idx = insertions[0][2]
                        best_pos = insertions[0][1]
                        best_route_length = route_lengths[best_route_idx]

                if best_customer is None:
                    best_customer = next(iter(unassigned))
                    best_new_max = float('inf')
                    best_route_idx = -1
                    best_pos = -1
                    best_route_length = float('inf')
                    for r_idx, route in enumerate(routes):
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            increase = distance_matrix[prev, best_customer] + distance_matrix[best_customer, nxt] - distance_matrix[prev, nxt]
                            new_len = route_lengths[r_idx] + increase
                            new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                            if new_max < best_new_max or (new_max == best_new_max and route_lengths[r_idx] < best_route_length):
                                best_new_max = new_max
                                best_route_idx = r_idx
                                best_pos = pos
                                best_route_length = route_lengths[r_idx]

                route = routes[best_route_idx]
                route.insert(best_pos, best_customer)
                route_lengths[best_route_idx] = compute_route_length(route)
                unassigned.remove(best_customer)
                report_best_vrp(routes)

            # Local search after repair
            local_search(routes, route_lengths)
            current_max = max(route_lengths)
            if current_max < best_max:
                best_max = current_max
                best_routes = [route[:] for route in routes]

    if best_routes is None:
        best_routes = routes
    return best_routes