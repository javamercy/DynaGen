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

    # Regret insertion construction
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
            if regret > best_regret or (regret == best_regret and (best_customer is None or best < best_delta)):
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

    current_max = max(route_distances)
    max_iter = n * n * 2
    iter_count = 0

    # Local search helpers
    def try_2opt():
        nonlocal improved, current_max
        for r_idx, route in enumerate(routes):
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = 0.0
                    for k in range(len(new_route)-1):
                        new_dist += dist[new_route[k], new_route[k+1]]
                    if new_dist < route_distances[r_idx]:
                        new_max = max(route_distances[:r_idx] + [new_dist] + route_distances[r_idx+1:])
                        if new_max < current_max:
                            routes[r_idx] = new_route
                            route_distances[r_idx] = new_dist
                            current_max = new_max
                            improved = True
                            try:
                                report_best_vrp(routes)
                            except NameError:
                                pass
                            return True
        return False

    def try_swap():
        nonlocal improved, current_max
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
                    ri_route = routes[ri]
                    rj_route = routes[rj]
                    delta_ri_rem = dist[ri_route[pos_i-1], i] + dist[i, ri_route[pos_i+1]] - dist[ri_route[pos_i-1], ri_route[pos_i+1]]
                    delta_rj_rem = dist[rj_route[pos_j-1], j] + dist[j, rj_route[pos_j+1]] - dist[rj_route[pos_j-1], rj_route[pos_j+1]]
                    delta_ri_add = dist[ri_route[pos_i-1], j] + dist[j, ri_route[pos_i+1]] - dist[ri_route[pos_i-1], ri_route[pos_i+1]]
                    delta_rj_add = dist[rj_route[pos_j-1], i] + dist[i, rj_route[pos_j+1]] - dist[rj_route[pos_j-1], rj_route[pos_j+1]]
                    new_ri_dist = route_distances[ri] - delta_ri_rem + delta_ri_add
                    new_rj_dist = route_distances[rj] - delta_rj_rem + delta_rj_add
                    new_max = max(route_distances[:ri] + [new_ri_dist] + route_distances[ri+1:rj] + [new_rj_dist] + route_distances[rj+1:])
                    if new_max < current_max:
                        ri_route[pos_i] = j
                        rj_route[pos_j] = i
                        route_distances[ri] = new_ri_dist
                        route_distances[rj] = new_rj_dist
                        current_max = new_max
                        improved = True
                        try:
                            report_best_vrp(routes)
                        except NameError:
                            pass
                        return True
        return False

    def try_relocate():
        nonlocal improved, current_max
        for cust in range(1, n):
            src_idx = None
            src_pos = None
            for idx, route in enumerate(routes):
                if cust in route:
                    src_idx = idx
                    src_pos = route.index(cust)
                    break
            if src_idx is None:
                continue
            src_route = routes[src_idx]
            prev = src_route[src_pos-1]
            nxt = src_route[src_pos+1]
            delta_rem = dist[prev, cust] + dist[cust, nxt] - dist[prev, nxt]
            new_src_dist = route_distances[src_idx] - delta_rem
            for dst_idx, dst_route in enumerate(routes):
                if dst_idx == src_idx:
                    continue
                for tpos in range(1, len(dst_route)):
                    tprev = dst_route[tpos-1]
                    tnxt = dst_route[tpos]
                    delta_add = dist[tprev, cust] + dist[cust, tnxt] - dist[tprev, tnxt]
                    new_dst_dist = route_distances[dst_idx] + delta_add
                    new_max = max(route_distances[:src_idx] + [new_src_dist] + route_distances[src_idx+1:dst_idx] + [new_dst_dist] + route_distances[dst_idx+1:])
                    if new_max < current_max:
                        src_route.pop(src_pos)
                        route_distances[src_idx] = new_src_dist
                        dst_route.insert(tpos, cust)
                        route_distances[dst_idx] = new_dst_dist
                        current_max = new_max
                        improved = True
                        try:
                            report_best_vrp(routes)
                        except NameError:
                            pass
                        return True
        return False

    while iter_count < max_iter:
        improved = False
        if try_2opt():
            iter_count += 1
            continue
        if try_swap():
            iter_count += 1
            continue
        if try_relocate():
            iter_count += 1
            continue
        break

    return [list(route) for route in routes]