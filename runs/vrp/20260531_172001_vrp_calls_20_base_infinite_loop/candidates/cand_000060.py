import numpy as np
import random
import time

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]
    
    start_time = time.time()
    max_time = 170
    
    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    def max_route_len(routes):
        return max(route_length(r) for r in routes)
    
    # ---- Construction: TSP tour via nearest neighbor + 2-opt ----
    # Build a tour starting and ending at depot (0)
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        next_node = min(unvisited, key=lambda x: distance_matrix[current, x])
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    tour.append(0)
    
    # Improve tour with 2-opt (fixed iterations)
    improved = True
    iters = 0
    max_iters = n * n
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(1, len(tour)-2):
            for k in range(i+1, len(tour)-1):
                new_tour = tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
                if route_length(new_tour) < route_length(tour) - 1e-12:
                    tour = new_tour
                    improved = True
                    break
            if improved:
                break
    
    # ---- Split TSP tour into truck_count routes using DP - min max distance ----
    # tour includes both depot at start and end; we need positions of customers (index 1..n-1 in tour)
    cust_indices = [i for i, node in enumerate(tour) if node != 0]  # indices of customers in tour
    # Precompute segment distances: dist[i][j] = distance from tour[cust_indices[i]] to tour[cust_indices[j]] via depot? No, we need route distance if we take a contiguous segment of tour and add depot at ends?
    # Actually, we want to cut the tour into segments, each segment will be a route starting and ending at depot.
    # A segment from position a to b in the tour (customers only) corresponds to route: 0 -> tour[a] -> ... -> tour[b] -> 0
    # The distance of such route is: dist(0, tour[a]) + distance along segment + dist(tour[b], 0).
    # So we need prefix sums along the customer sequence.
    cust_seq = [tour[i] for i in range(1, len(tour)-1)]  # customers in order (excluding depot at ends)
    k = len(cust_seq)  # number of customers
    if k == 0:
        # no customers
        routes = [[0, 0] for _ in range(truck_count)]
        return routes
    # Precompute prefix distances from start of customer sequence (including start from depot? We'll compute segment distances directly)
    # Let pref[i] = distance from depot to cust_seq[0] + sum of edges between cust_seq[0]..cust_seq[i-1] (i>=1), but we need distance of segment from a to b inclusive.
    # We'll compute seg_dist[i][j] for i <= j: distance from cust_seq[i] to cust_seq[j] (including edges between consecutive customers) + depot edges at ends.
    # But better: compute prefix distances from the first customer to others.
    # Compute distance along path from depot to first customer:
    depot_to_first = distance_matrix[0, cust_seq[0]]
    cum = [0.0] * (k + 1)  # cum[0] = 0; cum[i] = distance from start of customer seq (depot) to cust_seq[i-1]?
    # Actually define cum[i] = total distance from depot to customer at index i-1 inclusive? Let's define cum[i] = distance from depot to cust_seq[i-1] along the tour?
    # Simplify: Let edge_dist[i] = distance_matrix[cust_seq[i-1], cust_seq[i]] for i=1..k-1, with special first.
    # We'll compute segment distance for a contiguous block from l to r (0-indexed) as:
    # distance from depot to cust_seq[l] + sum of edges between cust_seq[l]..cust_seq[r] + distance from cust_seq[r] to depot.
    # We can precompute prefix sums of edges between customers.
    edge_between = [0.0] * (k - 1)
    for i in range(k-1):
        edge_between[i] = distance_matrix[cust_seq[i], cust_seq[i+1]]
    prefix_edges = [0.0] * (k + 1)  # prefix_edges[0]=0, prefix_edges[i]=sum edges from 0 to i-1
    for i in range(1, k):
        prefix_edges[i] = prefix_edges[i-1] + edge_between[i-1]
    prefix_edges[k] = prefix_edges[k-1]  # doesn't matter
    
    def segment_dist(l, r):
        # l and r are indices in cust_seq, inclusive
        # Distance of route: depot -> cust_seq[l] -> ... -> cust_seq[r] -> depot
        # distance = dist(depot, first) + sum edges between + dist(last, depot)
        d = distance_matrix[0, cust_seq[l]]
        d += (prefix_edges[r] - prefix_edges[l])  # sum of edges from l to r (if l <= r, edge between l and l+1 is in prefix_edges[l+1]-prefix_edges[l])
        d += distance_matrix[cust_seq[r], 0]
        return d
    
    # DP: dp[i][t] = minimal possible max distance for first i customers (indices 0..i-1) using t routes
    # Initialize large
    INF = 1e100
    dp = [[INF] * (truck_count + 1) for _ in range(k + 1)]
    # backtrack: best previous split point and route max
    prev = [[None] * (truck_count + 1) for _ in range(k + 1)]
    dp[0][0] = 0.0
    for i in range(1, k + 1):
        for t in range(1, min(truck_count, i) + 1):
            # try all j from 0 to i-1 as split point (j customers in first t-1 routes, then customers j..i-1 in t-th route)
            for j in range(0, i):
                if dp[j][t-1] >= INF:
                    continue
                seg_dist_val = segment_dist(j, i-1)
                cand = max(dp[j][t-1], seg_dist_val)
                if cand < dp[i][t]:
                    dp[i][t] = cand
                    prev[i][t] = (j, seg_dist_val)
    # We need exactly truck_count routes, but if k < truck_count, we have empty routes. Use dp[k][min(truck_count, k)]
    optimal_max = INF
    best_t = min(truck_count, k)
    # For feasibility, we must have at most truck_count routes; if k < truck_count, we can use k non-empty routes + (truck_count - k) empty routes
    # Our DP only up to min(truck_count, k) routes. If k < truck_count, we can fill empty routes later.
    if truck_count > k:
        # We'll use all k customers in k routes (one customer each) plus empty routes
        # The DP with t=k routes will give minimal max distance for k routes
        pass  # t = k is allowed
    else:
        # t = truck_count
        pass
    t_used = min(truck_count, k)
    if dp[k][t_used] >= INF:
        # fallback: just each customer in its own route (if possible)
        routes = [[0, cust, 0] for cust in range(1, n)] + [[0, 0]] * (truck_count - (n-1))
        return routes
    # Reconstruct routes
    routes = []
    cur_i = k
    cur_t = t_used
    while cur_i > 0 and cur_t > 0:
        j, seg_val = prev[cur_i][cur_t]
        # segment from j to cur_i-1
        seg_custs = cust_seq[j:cur_i]
        route = [0] + seg_custs + [0]
        routes.append(route)
        cur_i = j
        cur_t -= 1
    # routes list is in reverse order
    routes.reverse()
    # add empty routes if needed
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    best_max = max_route_len(routes)
    best_routes = [r[:] for r in routes]
    report_best_vrp(routes)
    
    # ---- Simulated Annealing Improvement ----
    current_routes = [r[:] for r in routes]
    current_max = best_max
    
    # SA parameters
    T_start = 10.0
    T_end = 0.01
    alpha = 0.99
    max_iter_total = n * truck_count * 5  # bound
    stagnation_threshold = max(10, n // 5)
    stagnation_count = 0
    
    iter_count = 0
    T = T_start
    while iter_count < max_iter_total:
        if time.time() - start_time > max_time:
            break
        iter_count += 1
        # Generate neighbor by random move: inter-route relocate or intra-2opt
        move_type = random.choice(['relocate', '2opt'])
        if move_type == 'relocate':
            # Choose a random customer (not depot) from a random route (min length 2)
            non_empty = [i for i, r in enumerate(current_routes) if len(r) > 2]
            if not non_empty:
                continue
            route_idx = random.choice(non_empty)
            route = current_routes[route_idx]
            # pick a customer (exclude depot)
            cust_pos = random.randrange(1, len(route)-1)
            cust = route[cust_pos]
            # remove customer
            new_route = route[:cust_pos] + route[cust_pos+1:]
            # choose a target route (any route, possibly same? but same route would be weird; better different)
            target_idx = random.randrange(truck_count)
            if target_idx == route_idx:
                # cannot reinsert into same route in this move; choose another
                continue
            target_route = current_routes[target_idx]
            # choose insertion position (1 to len(target_route))
            if len(target_route) == 2:
                # only one position: after depot
                ins_pos = 1
            else:
                ins_pos = random.randrange(1, len(target_route))
            new_target = target_route[:ins_pos] + [cust] + target_route[ins_pos:]
            # compute new max
            new_routes = [r[:] for r in current_routes]
            new_routes[route_idx] = new_route
            new_routes[target_idx] = new_target
            new_max = max_route_len(new_routes)
            delta = new_max - current_max
        else:  # 2-opt within a single route
            # choose a random route with at least 4 customers (so we can reverse a segment)
            long_routes = [i for i, r in enumerate(current_routes) if len(r) >= 4]
            if not long_routes:
                continue
            route_idx = random.choice(long_routes)
            route = current_routes[route_idx]
            # choose i and k randomly, 1 <= i < k <= len-2
            i = random.randrange(1, len(route)-2)
            k = random.randrange(i+1, len(route)-1)
            new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
            new_routes = [r[:] for r in current_routes]
            new_routes[route_idx] = new_route
            new_max = max_route_len(new_routes)
            delta = new_max - current_max
        
        # Accept or reject
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-12)):
            current_routes = new_routes
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in current_routes]
                report_best_vrp(best_routes)
                stagnation_count = 0
            else:
                stagnation_count += 1
        else:
            stagnation_count += 1
        
        # Adaptive restart if stagnation
        if stagnation_count >= stagnation_threshold:
            # Perturb: move a fraction of customers to random positions (from adaptive perturbation in parents)
            # perturb up to 25% customers
            perturb_size = min(max(1, n // 4), n-1)
            customers = list(range(1, n))
            random.shuffle(customers)
            for cust in customers[:perturb_size]:
                # remove from current route
                for r_idx, route in enumerate(current_routes):
                    if cust in route:
                        idx = route.index(cust)
                        if idx != 0 and idx != len(route)-1:
                            current_routes[r_idx] = route[:idx] + route[idx+1:]
                            break
                # insert into random route
                target_idx = random.randrange(truck_count)
                target_route = current_routes[target_idx]
                if len(target_route) == 2:
                    ins_pos = 1
                else:
                    ins_pos = random.randrange(1, len(target_route))
                current_routes[target_idx] = target_route[:ins_pos] + [cust] + target_route[ins_pos:]
            current_max = max_route_len(current_routes)
            stagnation_count = 0
            # Reset temperature to a higher value? maybe not.
        
        # Cool down
        T = max(T_end, T * alpha)
    
    if best_routes is None:
        return routes
    return best_routes