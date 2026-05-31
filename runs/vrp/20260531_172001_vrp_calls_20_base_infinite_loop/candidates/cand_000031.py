import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    def max_route_len(routes):
        return max(route_length(r) for r in routes)

    def best_insertion(cust, routes):
        best_cost = float('inf')
        best_r_idx = None
        best_pos = None
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                prev = route[pos-1]
                nxt = route[pos]
                cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                if cost < best_cost:
                    best_cost = cost
                    best_r_idx = r_idx
                    best_pos = pos
        return best_r_idx, best_pos, best_cost

    best_routes = None
    best_max = float('inf')
    max_attempts = max(1, n // 5)

    for attempt in range(max_attempts):
        # Construction with deterministic regret
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = list(range(1, n))
        while unassigned:
            best_regret = -float('inf')
            best_cust = None
            best_insert = None
            for cust in unassigned:
                costs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        costs.append((cost, r_idx, pos))
                costs.sort(key=lambda x: x[0])
                best_cost_cust = costs[0][0]
                second_cost_cust = costs[1][0] if len(costs) > 1 else best_cost_cust + 1e9
                regret = second_cost_cust - best_cost_cust
                if regret > best_regret or (regret == best_regret and cust < best_cust):
                    best_regret = regret
                    best_cust = cust
                    best_insert = costs[0]
            r_idx, pos = best_insert[1], best_insert[2]
            routes[r_idx].insert(pos, best_cust)
            unassigned.remove(best_cust)

        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

        # Variable Neighborhood Descent (VND)
        max_outer_iter = n * truck_count
        outer_iter = 0
        while outer_iter < max_outer_iter:
            improved = False
            outer_iter += 1

            # Inter-route relocate from longest route
            lengths = [route_length(r) for r in routes]
            max_idx = np.argmax(lengths)
            max_route = routes[max_idx]
            if len(max_route) > 2:
                best_delta = 0
                best_move = None
                for cust in max_route[1:-1]:
                    new_max_route = [x for x in max_route if x != cust]
                    new_max_len = route_length(new_max_route)
                    for r_idx in range(truck_count):
                        if r_idx == max_idx:
                            continue
                        other_route = routes[r_idx]
                        for pos in range(1, len(other_route)):
                            new_other = other_route[:pos] + [cust] + other_route[pos:]
                            new_other_len = route_length(new_other)
                            new_max_candidate = max(new_max_len, new_other_len, *[lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)])
                            if new_max_candidate < current_max:
                                delta = current_max - new_max_candidate
                                if delta > best_delta:
                                    best_delta = delta
                                    best_move = (cust, max_idx, r_idx, pos)
                if best_move:
                    cust, from_idx, to_idx, pos = best_move
                    routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                    routes[to_idx].insert(pos, cust)
                    current_max -= best_delta
                    improved = True
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)

            # Inter-route swap between longest and other routes
            lengths = [route_length(r) for r in routes]
            max_idx = np.argmax(lengths)
            max_route = routes[max_idx]
            for r_idx in range(truck_count):
                if r_idx == max_idx:
                    continue
                other_route = routes[r_idx]
                if len(max_route) <= 2 or len(other_route) <= 2:
                    continue
                best_delta = 0
                best_swap = None
                for i, cust1 in enumerate(max_route[1:-1], start=1):
                    for j, cust2 in enumerate(other_route[1:-1], start=1):
                        # swap cust1 and cust2
                        new_max = max_route[:i] + [cust2] + max_route[i+1:]
                        new_other = other_route[:j] + [cust1] + other_route[j+1:]
                        new_max_len = route_length(new_max)
                        new_other_len = route_length(new_other)
                        new_max_candidate = max(new_max_len, new_other_len, *[lengths[k] for k in range(truck_count) if k not in (max_idx, r_idx)])
                        if new_max_candidate < current_max:
                            delta = current_max - new_max_candidate
                            if delta > best_delta:
                                best_delta = delta
                                best_swap = (i, j, cust1, cust2, max_idx, r_idx)
                if best_swap:
                    i, j, cust1, cust2, from_idx, to_idx = best_swap
                    routes[from_idx][i] = cust2
                    routes[to_idx][j] = cust1
                    current_max -= best_delta
                    improved = True
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)

            # Intra-route 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for k in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                        new_len = route_length(new_route)
                        if new_len < route_length(route):
                            route[:] = new_route
                            improved = True
                            new_max = max_route_len(routes)
                            if new_max < current_max:
                                current_max = new_max
                                if current_max < best_max:
                                    best_max = current_max
                                    best_routes = [r[:] for r in routes]
                                    report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break

            # Intra-route Or-opt (move sequence of 1,2,3 customers)
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 4:
                    continue
                old_len = route_length(route)
                for seq_len in [1, 2, 3]:
                    for start in range(1, len(route)-seq_len):
                        seq = route[start:start+seq_len]
                        # remove seq
                        temp = route[:start] + route[start+seq_len:]
                        for insert_pos in range(1, len(temp)):
                            new_route = temp[:insert_pos] + seq + temp[insert_pos:]
                            new_len = route_length(new_route)
                            if new_len < old_len:
                                route[:] = new_route
                                improved = True
                                new_max = max_route_len(routes)
                                if new_max < current_max:
                                    current_max = new_max
                                    if current_max < best_max:
                                        best_max = current_max
                                        best_routes = [r[:] for r in routes]
                                        report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            if not improved:
                break  # exit VND

        # Ruin and recreate perturb (if local search stagnated)
        # Remove 20% of customers and reinsert with regret
        customers = list(range(1, n))
        random.shuffle(customers)
        num_remove = max(1, n // 5)
        removed = customers[:num_remove]
        for cust in removed:
            for r_idx, route in enumerate(routes):
                if cust in route:
                    route.remove(cust)
                    break
        # Reinsert removed using deterministic regret
        while removed:
            best_regret = -float('inf')
            best_cust = None
            best_insert = None
            for cust in removed:
                costs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        costs.append((cost, r_idx, pos))
                costs.sort(key=lambda x: x[0])
                best_cost_cust = costs[0][0]
                second_cost_cust = costs[1][0] if len(costs) > 1 else best_cost_cust + 1e9
                regret = second_cost_cust - best_cost_cust
                if regret > best_regret or (regret == best_regret and cust < best_cust):
                    best_regret = regret
                    best_cust = cust
                    best_insert = costs[0]
            r_idx, pos = best_insert[1], best_insert[2]
            routes[r_idx].insert(pos, best_cust)
            removed.remove(best_cust)

        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

    if best_routes is None:
        best_routes = routes
    return best_routes