import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = len(customers)
    if m == 0:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count >= m:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    # Phase 1: Build giant tour using deterministic savings
    chains = {c: [c] for c in customers}
    endpoints = {c: (c, c) for c in customers}
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                savings.append((-s, i, j))
    savings.sort()

    for _, i, j in savings:
        if i not in chains or j not in chains:
            continue
        ci = chains[i]
        cj = chains[j]
        if ci is cj:
            continue
        fi, li = endpoints[i]
        fj, lj = endpoints[j]
        new_chain = None
        new_first = None
        new_last = None
        if li == i and fj == j:
            new_chain = ci + cj
            new_first = fi
            new_last = lj
        elif lj == j and fi == i:
            new_chain = cj + ci
            new_first = fj
            new_last = li
        else:
            continue
        for cust in new_chain:
            chains[cust] = new_chain
            endpoints[cust] = (new_first, new_last)

    giant_tour = None
    for cust in customers:
        if len(chains[cust]) == m:
            giant_tour = chains[cust]
            break
    if giant_tour is None:
        giant_tour = []
        visited = set()
        for cust in customers:
            if cust not in visited:
                chain = chains[cust]
                giant_tour.extend(chain)
                visited.update(chain)

    # Phase 2: DP split
    def split_tour(tour, K):
        m = len(tour)
        if m == 0:
            return [[0, 0] for _ in range(K)], 0.0
        d0 = np.array([distance_matrix[0, c] for c in tour])
        dlast = np.array([distance_matrix[c, 0] for c in tour])
        pref_between = np.zeros(m + 1)
        for i in range(1, m):
            pref_between[i+1] = pref_between[i] + distance_matrix[tour[i-1], tour[i]]
        INF = 1e100
        dp = np.full((K+1, m+1), INF)
        prev = np.full((K+1, m+1), -1, dtype=int)
        dp[0, 0] = 0.0
        for k in range(1, K+1):
            for i in range(k, m+1):
                best = INF
                best_j = -1
                for j in range(k-1, i):
                    if dp[k-1, j] < INF:
                        seg = d0[j] + (pref_between[i] - pref_between[j+1]) + dlast[i-1]
                        max_val = max(dp[k-1, j], seg)
                        if max_val < best:
                            best = max_val
                            best_j = j
                dp[k, i] = best
                prev[k, i] = best_j
        best_max = dp[K, m]
        partitions = []
        k = K
        i = m
        while k > 0:
            j = prev[k, i]
            seg = tour[j:i]
            partitions.append(seg)
            i = j
            k -= 1
        partitions.reverse()
        routes = [[0] + seg + [0] for seg in partitions]
        while len(routes) < K:
            routes.append([0, 0])
        return routes, best_max

    initial_routes, best_max = split_tour(giant_tour, truck_count)
    best_routes = [list(r) for r in initial_routes]
    report_best_vrp(best_routes)

    # Phase 3: 2-opt on giant tour
    max_iter = min(20, m)
    for _ in range(max_iter):
        improved = False
        for a in range(m-2):
            for b in range(a+2, m):
                new_tour = giant_tour[:a+1] + giant_tour[a+1:b+1][::-1] + giant_tour[b+1:]
                old_dist = distance_matrix[0, giant_tour[0]] + sum(distance_matrix[giant_tour[i], giant_tour[i+1]] for i in range(m-1)) + distance_matrix[giant_tour[-1], 0]
                new_dist = distance_matrix[0, new_tour[0]] + sum(distance_matrix[new_tour[i], new_tour[i+1]] for i in range(m-1)) + distance_matrix[new_tour[-1], 0]
                if new_dist < old_dist - 1e-12:
                    giant_tour = new_tour
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
        new_routes, new_max = split_tour(giant_tour, truck_count)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in new_routes]
            report_best_vrp(best_routes)

    return best_routes