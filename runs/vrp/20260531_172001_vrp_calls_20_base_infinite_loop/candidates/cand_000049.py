import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def max_route_len(routes):
        return max(route_length(r) for r in routes)

    best_routes = None
    best_max = float('inf')
    attempts = max(1, n // 20)

    for _ in range(attempts):
        # Constructive phase: deterministic regret-2
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        while unassigned:
            best_cust = None
            best_regret = -1
            best_cost = None
            best_route_idx = None
            best_pos = None
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
                regret = costs[1][0] - best_cost_cust if len(costs) > 1 else 1e9
                if regret > best_regret or (regret == best_regret and (best_cost is None or best_cost_cust < best_cost)):
                    best_regret = regret
                    best_cost = best_cost_cust
                    best_cust = cust
                    best_route_idx = costs[0][1]
                    best_pos = costs[0][2]
            routes[best_route_idx].insert(best_pos, best_cust)
            unassigned.remove(best_cust)

        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        # Local search: best-improvement on max
        improved = True
        max_iter = n * truck_count
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            current_max = max_route_len(routes)
            best_delta = 0
            best_move = None

            # Inter-route relocate
            for from_idx in range(truck_count):
                route = routes[from_idx]
                if len(route) <= 2:
                    continue
                for cust_idx in range(1, len(route)-1):
                    cust = route[cust_idx]
                    for to_idx in range(truck_count):
                        if to_idx == from_idx:
                            continue
                        other_route = routes[to_idx]
                        for pos in range(1, len(other_route)):
                            new_from = route[:cust_idx] + route[cust_idx+1:]
                            new_to = other_route[:pos] + [cust] + other_route[pos:]
                            lengths = []
                            for i in range(truck_count):
                                if i == from_idx:
                                    lengths.append(route_length(new_from))
                                elif i == to_idx:
                                    lengths.append(route_length(new_to))
                                else:
                                    lengths.append(route_length(routes[i]))
                            new_max = max(lengths)
                            delta = current_max - new_max
                            if delta > best_delta:
                                best_delta = delta
                                best_move = ('inter_relocate', from_idx, cust_idx, to_idx, pos)

            # Intra-route relocate
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                old_len = route_length(route)
                for cust_idx in range(1, len(route)-1):
                    cust = route[cust_idx]
                    for new_pos in range(1, len(route)):
                        if new_pos == cust_idx or new_pos == cust_idx+1:
                            continue
                        new_route = route[:cust_idx] + route[cust_idx+1:]
                        new_route = new_route[:new_pos] + [cust] + new_route[new_pos:]
                        new_len = route_length(new_route)
                        if new_len < old_len:
                            # compute overall max
                            lengths = [route_length(r) for r in routes]
                            lengths[r_idx] = new_len
                            new_max = max(lengths)
                            delta = current_max - new_max
                            if delta > best_delta:
                                best_delta = delta
                                best_move = ('intra_relocate', r_idx, cust_idx, new_pos)

            # Intra-route 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                old_len = route_length(route)
                for i in range(1, len(route)-2):
                    for k in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                        new_len = route_length(new_route)
                        if new_len < old_len:
                            lengths = [route_length(r) for r in routes]
                            lengths[r_idx] = new_len
                            new_max = max(lengths)
                            delta = current_max - new_max
                            if delta > best_delta:
                                best_delta = delta
                                best_move = ('2opt', r_idx, i, k)

            if best_move:
                improved = True
                move_type = best_move[0]
                if move_type == 'inter_relocate':
                    _, from_idx, cust_idx, to_idx, pos = best_move
                    cust = routes[from_idx][cust_idx]
                    routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                    routes[to_idx].insert(pos, cust)
                elif move_type == 'intra_relocate':
                    _, r_idx, cust_idx, new_pos = best_move
                    route = routes[r_idx]
                    cust = route[cust_idx]
                    route.pop(cust_idx)
                    if new_pos > cust_idx:
                        new_pos -= 1
                    route.insert(new_pos, cust)
                elif move_type == '2opt':
                    _, r_idx, i, k = best_move
                    routes[r_idx] = routes[r_idx][:i] + routes[r_idx][i:k+1][::-1] + routes[r_idx][k+1:]
                new_max = max_route_len(routes)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)

        # After local search of this attempt, update best if needed
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

    if best_routes is None:
        best_routes = routes
    return best_routes