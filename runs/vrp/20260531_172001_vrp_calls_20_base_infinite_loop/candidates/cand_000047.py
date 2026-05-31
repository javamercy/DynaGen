import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_len(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def max_len(routes):
        return max(route_len(r) for r in routes)

    # Construction
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))
    while unassigned:
        best_cust = None
        best_r_idx = None
        best_pos = None
        best_cost = float('inf')
        best_regret = -float('inf')
        for cust in unassigned:
            inserts = []
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    cost = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                    new_len = route_len(route) + cost
                    other_lens = [route_len(routes[i]) for i in range(truck_count) if i != r_idx]
                    new_max = max(new_len, *other_lens)
                    inserts.append((new_max, cost, r_idx, pos))
            inserts.sort(key=lambda x: (x[0], x[1]))
            best = inserts[0]
            second = inserts[1] if len(inserts) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
            regret = second[0] - best[0]
            if (best[0] < best_cost) or (best[0] == best_cost and regret > best_regret) or (best[0] == best_cost and regret == best_regret and cust < best_cust):
                best_cust = cust
                best_cost = best[0]
                best_regret = regret
                best_r_idx = best[2]
                best_pos = best[3]
        routes[best_r_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)

    best_routes = [r[:] for r in routes]
    best_max = max_len(routes)
    report_best_vrp(routes)

    # VND
    max_iter = n * 2
    for _ in range(max_iter):
        improved = False
        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for k in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                    if route_len(new_route) < route_len(route) - 1e-12:
                        routes[r_idx] = new_route
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Inter-route relocate from longest route
            lengths = [route_len(r) for r in routes]
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            max_route = routes[max_idx]
            if len(max_route) > 2:
                for cust in max_route[1:-1]:
                    for r_idx in range(truck_count):
                        if r_idx == max_idx:
                            continue
                        other = routes[r_idx]
                        for pos in range(1, len(other)):
                            new_max_route = [x for x in max_route if x != cust]
                            new_other = other[:pos] + [cust] + other[pos:]
                            new_lens = lengths[:]
                            new_lens[max_idx] = route_len(new_max_route)
                            new_lens[r_idx] = route_len(new_other)
                            new_max = max(new_lens)
                            if new_max < best_max - 1e-12:
                                routes[max_idx] = new_max_route
                                routes[r_idx] = new_other
                                best_max = new_max
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        if improved:
            best_routes = [r[:] for r in routes]
            best_max = max_len(routes)
            report_best_vrp(routes)
        else:
            break

    return best_routes