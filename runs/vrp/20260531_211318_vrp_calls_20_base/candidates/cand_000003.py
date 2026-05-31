import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    dist = distance_matrix
    customers = list(range(1, n))
    unassigned = set(customers)
    routes = [[0, 0] for _ in range(truck_count)]
    route_distances = [0.0] * truck_count

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

    # report initial
    try:
        report_best_vrp(routes)
    except NameError:
        pass

    # local search: inter-route swaps
    n_cust = n - 1
    max_iter = n_cust * n_cust
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
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
                    # remove i from ri
                    prev_i = route_ri[pos_i-1]
                    next_i = route_ri[pos_i+1]
                    delta_ri_rem = dist[prev_i, i] + dist[i, next_i] - dist[prev_i, next_i]
                    # remove j from rj
                    prev_j = route_rj[pos_j-1]
                    next_j = route_rj[pos_j+1]
                    delta_rj_rem = dist[prev_j, j] + dist[j, next_j] - dist[prev_j, next_j]
                    # insert j into ri at original pos_i
                    delta_ri_add = dist[prev_i, j] + dist[j, next_i] - dist[prev_i, next_i]
                    # insert i into rj at original pos_j
                    delta_rj_add = dist[prev_j, i] + dist[i, next_j] - dist[prev_j, next_j]
                    new_dist_ri = route_distances[ri] - delta_ri_rem + delta_ri_add
                    new_dist_rj = route_distances[rj] - delta_rj_rem + delta_rj_add
                    current_max = max(route_distances)
                    others_max = 0
                    for k in range(truck_count):
                        if k != ri and k != rj:
                            others_max = max(others_max, route_distances[k])
                    new_max = max(new_dist_ri, new_dist_rj, others_max)
                    if new_max < current_max:
                        # apply swap
                        route_ri.pop(pos_i)
                        route_rj.pop(pos_j)
                        route_ri.insert(pos_i, j)
                        route_rj.insert(pos_j, i)
                        route_distances[ri] = new_dist_ri
                        route_distances[rj] = new_dist_rj
                        improved = True
                        # report if better
                        try:
                            report_best_vrp(routes)
                        except NameError:
                            pass
        iter_count += 1

    return [list(route) for route in routes]