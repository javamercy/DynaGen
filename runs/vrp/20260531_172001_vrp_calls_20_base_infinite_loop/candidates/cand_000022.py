import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # Construction: regret insertion
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(customers)
    while unassigned:
        best_customer = None
        best_regret = -1.0
        best_route_idx = None
        best_pos = None
        best_ins_cost = None
        for cust in unassigned:
            costs = []
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    costs.append((cost, r_idx, pos))
            costs.sort(key=lambda x: x[0])
            best_cost = costs[0][0]
            second_cost = costs[1][0] if len(costs) > 1 else best_cost + 1e9
            regret = second_cost - best_cost
            if regret > best_regret or (regret == best_regret and (best_ins_cost is None or best_cost > best_ins_cost)):
                best_regret = regret
                best_customer = cust
                best_route_idx = costs[0][1]
                best_pos = costs[0][2]
                best_ins_cost = best_cost
            elif regret == best_regret and best_cost == best_ins_cost and cust < best_customer:
                best_customer = cust
                best_route_idx = costs[0][1]
                best_pos = costs[0][2]
                best_ins_cost = best_cost
        route = routes[best_route_idx]
        route.insert(best_pos, best_customer)
        unassigned.remove(best_customer)
    
    def route_len(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def compute_max():
        return max(route_len(r) for r in routes)
    
    current_max = compute_max()
    # Multi-neighborhood local search
    max_iter = n * truck_count * 2
    for iteration in range(max_iter):
        improved = False
        # Relocate: move one customer to another route
        # Try the move that gives largest reduction in max
        best_move = None
        best_delta = 0
        for from_idx in range(truck_count):
            route_from = routes[from_idx]
            if len(route_from) <= 2:
                continue
            for idx, cust in enumerate(route_from[1:-1], start=1):
                # Temporarily remove cust
                new_from = route_from[:idx] + route_from[idx+1:]
                new_from_len = route_len(new_from)
                for to_idx in range(truck_count):
                    if to_idx == from_idx:
                        continue
                    route_to = routes[to_idx]
                    for pos in range(1, len(route_to)):
                        new_to = route_to[:pos] + [cust] + route_to[pos:]
                        new_to_len = route_len(new_to)
                        # Compute new max
                        candidates = [new_from_len, new_to_len]
                        for i in range(truck_count):
                            if i not in (from_idx, to_idx):
                                candidates.append(route_len(routes[i]))
                        new_max = max(candidates)
                        if new_max < current_max:
                            delta = current_max - new_max
                            if delta > best_delta:
                                best_delta = delta
                                best_move = ('relocate', from_idx, idx, to_idx, pos, cust)
        # Swap: exchange customers between two routes
        for i in range(truck_count):
            for j in range(i+1, truck_count):
                route_i = routes[i]
                route_j = routes[j]
                if len(route_i) <= 2 or len(route_j) <= 2:
                    continue
                for p, cust_i in enumerate(route_i[1:-1], start=1):
                    for q, cust_j in enumerate(route_j[1:-1], start=1):
                        # Swap cust_i and cust_j
                        new_i = route_i[:p] + [cust_j] + route_i[p+1:]
                        new_j = route_j[:q] + [cust_i] + route_j[q+1:]
                        new_i_len = route_len(new_i)
                        new_j_len = route_len(new_j)
                        candidates = [new_i_len, new_j_len]
                        for k in range(truck_count):
                            if k not in (i, j):
                                candidates.append(route_len(routes[k]))
                        new_max = max(candidates)
                        if new_max < current_max:
                            delta = current_max - new_max
                            if delta > best_delta:
                                best_delta = delta
                                best_move = ('swap', i, p, j, q, cust_i, cust_j)
        if best_move:
            if best_move[0] == 'relocate':
                _, from_idx, idx, to_idx, pos, cust = best_move
                routes[from_idx] = routes[from_idx][:idx] + routes[from_idx][idx+1:]
                routes[to_idx].insert(pos, cust)
                current_max -= best_delta
                improved = True
            else:
                _, i, p, j, q, cust_i, cust_j = best_move
                routes[i] = routes[i][:p] + [cust_j] + routes[i][p+1:]
                routes[j] = routes[j][:q] + [cust_i] + routes[j][q+1:]
                current_max -= best_delta
                improved = True
            report_best_vrp(routes)
        else:
            # Intra-route 2-opt on each route
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                old_len = route_len(route)
                for i in range(1, len(route)-2):
                    for k in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                        new_len = route_len(new_route)
                        if new_len < old_len:
                            # Update route
                            route[:] = new_route
                            new_max = compute_max()
                            if new_max < current_max:
                                current_max = new_max
                                report_best_vrp(routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break
    # After local search, attempt a focused intensification on the worst route
    for _ in range(5):
        max_len = compute_max()
        worst_idx = max(range(truck_count), key=lambda i: route_len(routes[i]))
        worst_route = routes[worst_idx]
        if len(worst_route) <= 2:
            break
        # Try to relocate each customer from worst route to other routes using cheapest insertion
        # that reduces max
        best_move = None
        best_delta = 0
        for idx, cust in enumerate(worst_route[1:-1], start=1):
            new_worst = worst_route[:idx] + worst_route[idx+1:]
            new_worst_len = route_len(new_worst)
            for to_idx in range(truck_count):
                if to_idx == worst_idx:
                    continue
                other_route = routes[to_idx]
                # find best insertion position to minimize max
                for pos in range(1, len(other_route)):
                    new_other = other_route[:pos] + [cust] + other_route[pos:]
                    new_other_len = route_len(new_other)
                    candidates = [new_worst_len, new_other_len]
                    for i in range(truck_count):
                        if i not in (worst_idx, to_idx):
                            candidates.append(route_len(routes[i]))
                    new_max = max(candidates)
                    if new_max < max_len:
                        delta = max_len - new_max
                        if delta > best_delta:
                            best_delta = delta
                            best_move = (idx, to_idx, pos, cust)
        if best_move:
            idx, to_idx, pos, cust = best_move
            routes[worst_idx] = worst_route[:idx] + worst_route[idx+1:]
            routes[to_idx].insert(pos, cust)
            current_max -= best_delta
            report_best_vrp(routes)
        else:
            break
    return routes