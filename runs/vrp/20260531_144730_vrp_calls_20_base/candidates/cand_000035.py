import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 0:
        return []
    customer_count = n - 1
    if truck_count <= 0:
        return []
    if truck_count >= customer_count:
        routes = [[0, 0] for _ in range(truck_count)]
        for i in range(1, n):
            routes[i - 1] = [0, i, 0]
        return routes

    # Helper functions
    def route_dist(route):
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    def two_opt_route(route):
        if len(route) <= 3:
            return route[:]
        best = route[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best)-2):
                for j in range(i+1, len(best)-1):
                    if j - i == 1:
                        continue
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    if route_dist(new_route) < route_dist(best) - 1e-12:
                        best = new_route
                        improved = True
                        break
                if improved:
                    break
        return best

    # Build initial permutation via nearest neighbor
    perm = []
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        next_node = min(unvisited, key=lambda x: (distance_matrix[current, x], x))
        perm.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    # 2-opt improvement on full tour (including depot)
    full_tour = [0] + perm[:] + [0]
    best_tour = two_opt_route(full_tour)[1:-1]
    perm = best_tour
    m = len(perm)

    # Precompute segment distances for DP
    start_to_depot = np.array([distance_matrix[0, c] for c in perm])
    end_to_depot = np.array([distance_matrix[c, 0] for c in perm])
    cum_inter = np.zeros(m + 1)
    for i in range(1, m):
        cum_inter[i] = cum_inter[i-1] + distance_matrix[perm[i-1], perm[i]]
    cum_inter[m] = cum_inter[m-1]

    def seg_dist(l, r):
        if l > r:
            return 0.0
        return start_to_depot[l] + (cum_inter[r] - cum_inter[l]) + end_to_depot[r]

    def split_perm(perm):
        m = len(perm)
        K = truck_count
        INF = float('inf')
        dp = [[INF] * (m + 1) for _ in range(K + 1)]
        choice = [[-1] * (m + 1) for _ in range(K + 1)]
        dp[0][0] = 0.0
        for t in range(1, K + 1):
            for i in range(t, m + 1):
                best_val = INF
                best_j = -1
                for j in range(t - 1, i):
                    cand = max(dp[t-1][j], seg_dist(j, i-1))
                    if cand < best_val - 1e-12:
                        best_val = cand
                        best_j = j
                    elif abs(cand - best_val) < 1e-12 and j < best_j:
                        best_j = j
                dp[t][i] = best_val
                choice[t][i] = best_j
        routes = []
        i = m
        for t in range(K, 0, -1):
            j = choice[t][i]
            if j == i:
                routes.append([0, 0])
            else:
                route = [0] + perm[j:i] + [0]
                routes.append(route)
            i = j
        routes.reverse()
        return routes

    # Initial routes
    routes = split_perm(perm)
    for idx in range(truck_count):
        routes[idx] = two_opt_route(routes[idx])
    best_routes = [r[:] for r in routes]
    best_max = max_dist(best_routes)
    report_best_vrp(best_routes)

    # Ruin and recreate (adapted from cand_000017)
    max_iter_rr = customer_count * 2
    for _ in range(max_iter_rr):
        old_max = max_dist(routes)
        # Compute customer contributions
        contribs = []
        for ridx, route in enumerate(routes):
            if len(route) <= 2:
                continue
            for pos in range(1, len(route)-1):
                cust = route[pos]
                c = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos+1]]
                contribs.append((-c, cust, ridx, pos))
        if not contribs:
            break
        contribs.sort()
        ruin_size = max(1, customer_count // 10)
        to_ruin = [t[1] for t in contribs[:ruin_size]]
        saved_routes = [r[:] for r in routes]
        # Remove customers
        for cust in to_ruin:
            for ridx, route in enumerate(routes):
                if cust in route:
                    new_route = [x for x in route if x != cust]
                    if len(new_route) < 2:
                        new_route = [0, 0]
                    else:
                        if new_route[0] != 0:
                            new_route.insert(0, 0)
                        if new_route[-1] != 0:
                            new_route.append(0)
                    routes[ridx] = new_route
                    break
        # Reinsert with regret-2
        unrouted = set(to_ruin)
        current_routes = [r[:] for r in routes]
        while unrouted:
            best_cust = None
            best_regret = -1.0
            best_insert = (None, None, None)  # ridx, pos, cost
            for cust in sorted(unrouted):
                costs = []
                for ri, route in enumerate(current_routes):
                    best_cost = float('inf')
                    best_pos = -1
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        temp_routes = current_routes[:]
                        temp_routes[ri] = new_route
                        new_max = max_dist(temp_routes)
                        if new_max < best_cost - 1e-12:
                            best_cost = new_max
                            best_pos = pos
                        elif abs(new_max - best_cost) < 1e-12 and pos < best_pos:
                            best_pos = pos
                    costs.append((best_cost, best_pos, ri))
                costs.sort(key=lambda x: (x[0], x[2]))
                if len(costs) >= 2:
                    regret = costs[1][0] - costs[0][0]
                else:
                    regret = 0.0
                if regret > best_regret + 1e-12:
                    best_regret = regret
                    best_cust = cust
                    best_insert = (costs[0][2], costs[0][1], costs[0][0])
            ri, pos, _ = best_insert
            route = current_routes[ri]
            current_routes[ri] = route[:pos] + [best_cust] + route[pos:]
            unrouted.remove(best_cust)
        # Convert routes to permutation
        new_perm = []
        for route in current_routes:
            for node in route:
                if node != 0:
                    new_perm.append(node)
        # Ensure permutation covers all customers exactly once
        if len(new_perm) == customer_count and len(set(new_perm)) == customer_count:
            new_routes = split_perm(new_perm)
            for idx in range(truck_count):
                new_routes[idx] = two_opt_route(new_routes[idx])
            new_max = max_dist(new_routes)
            if new_max < best_max - 1e-12:
                best_max = new_max
                best_routes = [r[:] for r in new_routes]
                routes = new_routes
                perm = new_perm
                report_best_vrp(best_routes)
            else:
                routes = saved_routes
        else:
            routes = saved_routes

    # Permutation swap local search (adapted from cand_000015)
    max_iter_swap = customer_count * truck_count
    for _ in range(max_iter_swap):
        improved = False
        for i in range(m):
            for j in range(i+1, m):
                new_perm = perm[:]
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_routes = split_perm(new_perm)
                for idx in range(truck_count):
                    new_routes[idx] = two_opt_route(new_routes[idx])
                new_max = max_dist(new_routes)
                if new_max < best_max - 1e-12:
                    best_max = new_max
                    best_routes = [r[:] for r in new_routes]
                    perm = new_perm
                    improved = True
                    report_best_vrp(best_routes)
                    break
            if improved:
                break
        if not improved:
            break

    return best_routes