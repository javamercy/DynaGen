import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0] * truck_count

    def route_len(route):
        l = 0.0
        for i in range(len(route)-1):
            l += distance_matrix[route[i], route[i+1]]
        return l

    # update route_lengths
    for r in range(truck_count):
        route_lengths[r] = route_len(routes[r])

    # Construction: insert customers 1..n-1
    for cust in range(1, n):
        best_max = float('inf')
        best_route = 0
        best_pos = 0
        for r in range(truck_count):
            rt = routes[r]
            for p in range(1, len(rt)):
                prev = rt[p-1]
                nxt = rt[p]
                new_len = route_lengths[r] - distance_matrix[prev, nxt] + distance_matrix[prev, cust] + distance_matrix[cust, nxt]
                new_max = new_len
                for rr in range(truck_count):
                    if rr != r and route_lengths[rr] > new_max:
                        new_max = route_lengths[rr]
                if new_max < best_max or (new_max == best_max and (r < best_route or (r == best_route and p < best_pos))):
                    best_max = new_max
                    best_route = r
                    best_pos = p
        # insert
        rt = routes[best_route]
        rt.insert(best_pos, cust)
        route_lengths[best_route] = route_len(rt)

    def report_best_vrp(routes):
        pass  # external function

    report_best_vrp([list(r) for r in routes])
    current_max = max(route_lengths)

    # Local search: inter-route relocations
    max_passes = n  # finite bound
    for _ in range(max_passes):
        improved = False
        for r_from in range(truck_count):
            rt_from = routes[r_from]
            if len(rt_from) <= 2:
                continue
            # iterate over a copy of customer indices to avoid modification issues
            for idx in range(1, len(rt_from)-1):
                cust = rt_from[idx]
                # remember original state
                orig_rt_from = rt_from[:]
                orig_len_from = route_lengths[r_from]
                # remove customer
                new_rt_from = rt_from[:idx] + rt_from[idx+1:]
                new_len_from = route_len(new_rt_from)
                for r_to in range(truck_count):
                    if r_to == r_from:
                        continue
                    rt_to = routes[r_to]
                    for p in range(1, len(rt_to)):
                        new_rt_to = rt_to[:p] + [cust] + rt_to[p:]
                        new_len_to = route_len(new_rt_to)
                        new_max = max(new_len_from, new_len_to)
                        for rr in range(truck_count):
                            if rr != r_from and rr != r_to:
                                if route_lengths[rr] > new_max:
                                    new_max = route_lengths[rr]
                        if new_max < current_max:
                            # accept move
                            routes[r_from] = new_rt_from
                            routes[r_to] = new_rt_to
                            route_lengths[r_from] = new_len_from
                            route_lengths[r_to] = new_len_to
                            current_max = new_max
                            improved = True
                            report_best_vrp([list(r) for r in routes])
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return [list(r) for r in routes]