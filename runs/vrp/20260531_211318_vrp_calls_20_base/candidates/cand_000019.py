import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    dist = distance_matrix
    customers = list(range(1, n))
    unassigned = set(customers)
    routes = [[0, 0] for _ in range(truck_count)]
    route_distances = [0.0] * truck_count

    # Regret insertion construction (same as parent)
    while unassigned:
        best_customer = None
        best_regret = -float('inf')
        best_delta = None
        best_route_idx = None
        best_pos = None
        for cust in unassigned:
            deltas = []
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    delta = dist[prev, cust] + dist[cust, nxt] - dist[prev, nxt]
                    deltas.append((delta, r_idx, pos))
            if not deltas:
                continue
            deltas.sort(key=lambda x: x[0])
            best = deltas[0][0]
            if len(deltas) > 1:
                second = deltas[1][0]
                regret = second - best
            else:
                regret = float('inf')
            if regret > best_regret or (regret == best_regret and (best_customer is None or best > best_delta)):
                best_regret = regret
                best_customer = cust
                best_delta = best
                best_route_idx = deltas[0][1]
                best_pos = deltas[0][2]
        route = routes[best_route_idx]
        route_distances[best_route_idx] += best_delta
        route.insert(best_pos, best_customer)
        unassigned.remove(best_customer)

    try:
        report_best_vrp(routes)
    except NameError:
        pass

    # Simulated annealing parameters
    n_cust = n - 1
    max_iter = n_cust * n_cust * 2
    T_start = max(route_distances) * 0.2
    T_end = 0.001
    cooling_rate = (T_end / T_start) ** (1.0 / max_iter)
    T = T_start

    best_routes = [list(r) for r in routes]
    best_max = max(route_distances)

    for iteration in range(max_iter):
        improved = False
        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = 0
                    for k in range(len(new_route)-1):
                        new_dist += dist[new_route[k], new_route[k+1]]
                    delta = new_dist - route_distances[r_idx]
                    if delta < 0 or (T > 0 and math.exp(-delta / T) > np.random.random()):
                        routes[r_idx] = new_route
                        route_distances[r_idx] = new_dist
                        if new_dist < route_distances[r_idx]:
                            improved = True
                        break
                break
        # Inter-route swap
        for i in range(1, n):
            for j in range(i+1, n):
                ri = None
                rj = None
                pos_i = None
                pos_j = None
                for idx, route in enumerate(routes):
                    if i in route:
                        ri = idx
                        pos_i = route.index(i)
                    if j in route:
                        rj = idx
                        pos_j = route.index(j)
                if ri is not None and rj is not None and ri != rj:
                    route_ri = routes[ri]
                    route_rj = routes[rj]
                    prev_i = route_ri[pos_i-1]
                    next_i = route_ri[pos_i+1]
                    delta_ri_rem = dist[prev_i, i] + dist[i, next_i] - dist[prev_i, next_i]
                    prev_j = route_rj[pos_j-1]
                    next_j = route_rj[pos_j+1]
                    delta_rj_rem = dist[prev_j, j] + dist[j, next_j] - dist[prev_j, next_j]
                    delta_ri_add = dist[prev_i, j] + dist[j, next_i] - dist[prev_i, next_i]
                    delta_rj_add = dist[prev_j, i] + dist[i, next_j] - dist[prev_j, next_j]
                    new_dist_ri = route_distances[ri] - delta_ri_rem + delta_ri_add
                    new_dist_rj = route_distances[rj] - delta_rj_rem + delta_rj_add
                    current_max = max(route_distances)
                    others_max = max(route_distances[k] for k in range(truck_count) if k not in (ri, rj)) if truck_count > 2 else 0
                    new_max = max(new_dist_ri, new_dist_rj, others_max)
                    delta = new_max - current_max
                    if delta < 0 or (T > 0 and math.exp(-delta / T) > np.random.random()):
                        route_ri.pop(pos_i)
                        route_rj.pop(pos_j)
                        route_ri.insert(pos_i, j)
                        route_rj.insert(pos_j, i)
                        route_distances[ri] = new_dist_ri
                        route_distances[rj] = new_dist_rj
                        if new_max < current_max:
                            improved = True
                        break
            if improved:
                break
        # Inter-route relocate
        max_route_idx = max(range(truck_count), key=lambda r: route_distances[r])
        max_route = routes[max_route_idx]
        if len(max_route) > 2:
            customer_to_move = None
            best_delta_total = 0
            best_target_route = None
            best_pos = None
            current_max = max(route_distances)
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                prev = max_route[pos-1]
                nxt = max_route[pos+1]
                delta_rem = dist[prev, cust] + dist[cust, nxt] - dist[prev, nxt]
                for t_idx, t_route in enumerate(routes):
                    if t_idx == max_route_idx:
                        continue
                    for tpos in range(1, len(t_route)):
                        tprev = t_route[tpos-1]
                        tnxt = t_route[tpos]
                        delta_add = dist[tprev, cust] + dist[cust, tnxt] - dist[tprev, tnxt]
                        new_dist_max = route_distances[max_route_idx] - delta_rem
                        new_dist_t = route_distances[t_idx] + delta_add
                        other_max = max(route_distances[k] for k in range(truck_count) if k not in (max_route_idx, t_idx))
                        new_max = max(new_dist_max, new_dist_t, other_max)
                        delta = new_max - current_max
                        if delta < 0 or (T > 0 and math.exp(-delta / T) > np.random.random()):
                            if customer_to_move is None or (delta < 0 and new_max < current_max):
                                customer_to_move = cust
                                best_delta_rem = delta_rem
                                best_delta_add = delta_add
                                best_target_route = t_idx
                                best_pos = tpos
            if customer_to_move is not None:
                routes[max_route_idx].pop(pos)
                route_distances[max_route_idx] -= best_delta_rem
                routes[best_target_route].insert(best_pos, customer_to_move)
                route_distances[best_target_route] += best_delta_add
                new_max = max(route_distances)
                if new_max < current_max:
                    improved = True
        # Update best solution
        current_max = max(route_distances)
        if current_max < best_max:
            best_routes = [list(r) for r in routes]
            best_max = current_max
            try:
                report_best_vrp(routes)
            except NameError:
                pass
        # Cooling
        T *= cooling_rate
        if T < T_end:
            T = T_end
        if not improved:
            break

    return best_routes