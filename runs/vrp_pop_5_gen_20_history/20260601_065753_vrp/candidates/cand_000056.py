import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def route_length(route):
        if len(route) <= 1:
            return 0.0
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    def compute_max_length(routes):
        return max(route_length(r) for r in routes)

    best_max = float('inf')
    best_routes = None

    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = compute_max_length(routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]

    def construct(alpha):
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0.0] * truck_count
        unassigned = set(customers)
        while unassigned:
            candidates = []
            best_cost = float('inf')
            for cust in list(unassigned):
                for t in range(truck_count):
                    route = routes[t]
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_len = route_length(new_route)
                        new_max = new_len
                        for k in range(truck_count):
                            if k != t:
                                new_max = max(new_max, lengths[k])
                        cost = new_max
                        if cost < best_cost:
                            best_cost = cost
                        candidates.append((cost, cust, t, pos))
            # Filter RCL: costs <= best_cost * (1+alpha)
            rcl = [c for c in candidates if c[0] <= best_cost * (1 + alpha)]
            # Tie-break by cost then customer index
            rcl.sort(key=lambda x: (x[0], x[1]))
            # Select randomly from RCL
            chosen = random.choice(rcl)
            cost, cust, t, pos = chosen
            routes[t] = routes[t][:pos] + [cust] + routes[t][pos:]
            lengths[t] = route_length(routes[t])
            unassigned.remove(cust)
        return routes, lengths

    def local_search(routes, lengths):
        improved = True
        max_iter_ls = 50 * (n + truck_count)
        it = 0
        while improved and it < max_iter_ls:
            improved = False
            it += 1
            # 2-opt intra-route
            for t in range(truck_count):
                route = routes[t]
                if len(route) <= 3:
                    continue
                best_delta = 0
                best_ij = None
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_len = route_length(new_route)
                        delta = new_len - lengths[t]
                        new_max = max(new_len, max(lengths[:t] + lengths[t+1:], default=0))
                        if new_max < max(lengths):
                            if delta < best_delta:
                                best_delta = delta
                                best_ij = (i, j)
                if best_ij is not None:
                    i, j = best_ij
                    route_new = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    routes[t] = route_new
                    lengths[t] = route_length(route_new)
                    improved = True
            # Relocate inter-route
            for t1 in range(truck_count):
                route1 = routes[t1]
                if len(route1) <= 2:
                    continue
                for i in range(1, len(route1)-1):
                    cust = route1[i]
                    new_route1 = route1[:i] + route1[i+1:]
                    len1_new = route_length(new_route1)
                    for t2 in range(truck_count):
                        if t1 == t2:
                            continue
                        route2 = routes[t2]
                        for j in range(1, len(route2)):
                            new_route2 = route2[:j] + [cust] + route2[j:]
                            len2_new = route_length(new_route2)
                            new_max = max(len1_new, len2_new,
                                          max(lengths[:t1] + lengths[t1+1:t2] + lengths[t2+1:], default=0))
                            if new_max < max(lengths):
                                routes[t1] = new_route1
                                routes[t2] = new_route2
                                lengths[t1] = len1_new
                                lengths[t2] = len2_new
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Swap inter-route
            for t1 in range(truck_count):
                route1 = routes[t1]
                if len(route1) <= 2:
                    continue
                for i in range(1, len(route1)-1):
                    cust1 = route1[i]
                    for t2 in range(t1+1, truck_count):
                        route2 = routes[t2]
                        if len(route2) <= 2:
                            continue
                        for j in range(1, len(route2)-1):
                            cust2 = route2[j]
                            new_route1 = route1[:i] + [cust2] + route1[i+1:]
                            new_route2 = route2[:j] + [cust1] + route2[j+1:]
                            len1_new = route_length(new_route1)
                            len2_new = route_length(new_route2)
                            new_max = max(len1_new, len2_new,
                                          max(lengths[:t1] + lengths[t1+1:t2] + lengths[t2+1:], default=0))
                            if new_max < max(lengths):
                                routes[t1] = new_route1
                                routes[t2] = new_route2
                                lengths[t1] = len1_new
                                lengths[t2] = len2_new
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes, lengths

    # Initial deterministic construction (alpha=0)
    routes_init, lengths_init = construct(0.0)
    report_best_vrp(routes_init)

    # Multi-start iterations
    num_iter = 50 * n
    for _ in range(num_iter):
        alpha = random.uniform(0.0, 0.3)
        routes, lengths = construct(alpha)
        routes, lengths = local_search(routes, lengths)
        report_best_vrp(routes)

    return best_routes