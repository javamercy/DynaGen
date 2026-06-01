import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    # Step 1: Build giant TSP tour using farthest-first insertion
    # Sort customers by distance to depot descending, tie break by index
    sorted_cust = sorted(customers, key=lambda c: (-distance_matrix[0, c], c))
    
    # Initialize tour as [0, 0]
    tour = [0, 0]
    
    def tour_distance(t):
        d = 0.0
        for i in range(len(t)-1):
            d += distance_matrix[t[i], t[i+1]]
        return d
    
    for cust in sorted_cust:
        best_increase = float('inf')
        best_pos = -1
        current_total = tour_distance(tour)
        # Insert at positions 1 to len(tour)-1 (between nodes)
        for pos in range(1, len(tour)):
            # Increase if inserting cust between tour[pos-1] and tour[pos]
            increase = distance_matrix[tour[pos-1], cust] + distance_matrix[cust, tour[pos]] - distance_matrix[tour[pos-1], tour[pos]]
            if increase < best_increase:
                best_increase = increase
                best_pos = pos
            # tie: earlier position first (pos smaller)
        tour = tour[:best_pos] + [cust] + tour[best_pos:]
    
    # tour now is a TSP tour starting and ending at 0, with all customers in between
    # Remove the trailing 0 for sequence of customers
    seq = tour[1:-1]  # list of customer indices in order
    m = len(seq)
    
    # Step 2: Precompute segment distances
    # segment_dist[i][j] = distance of route from 0 to seq[i] then along seq to seq[j] then back to 0
    segment_dist = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i, m):
            d = distance_matrix[0, seq[i]]
            for k in range(i, j):
                d += distance_matrix[seq[k], seq[k+1]]
            d += distance_matrix[seq[j], 0]
            segment_dist[i][j] = d
    
    # Step 3: DP for minimum max distance partition
    # dp[k][i] = min possible max distance when splitting first i customers (seq[0:i]) into k routes
    # We'll use 1-indexed: i from 0 to m, k from 1 to truck_count
    INF = float('inf')
    dp = [[INF] * (m+1) for _ in range(truck_count+1)]
    split = [[-1] * (m+1) for _ in range(truck_count+1)]  # store split point for reconstruction
    
    # Base: k=1
    for i in range(1, m+1):
        dp[1][i] = segment_dist[0][i-1]
        split[1][i] = 0  # split at start
    
    # Fill dp for k>=2
    for k in range(2, truck_count+1):
        for i in range(k, m+1):  # need at least k customers
            best_val = INF
            best_j = -1
            # j is the number of customers assigned to first k-1 routes, so j from k-1 to i-1
            for j in range(k-1, i):
                val = max(dp[k-1][j], segment_dist[j][i-1])
                if val < best_val:
                    best_val = val
                    best_j = j
                # tie: earliest j (smaller) - already satisfied because we loop increasing j
            dp[k][i] = best_val
            split[k][i] = best_j
    
    # Reconstruct routes
    routes = []
    curr_k = truck_count
    curr_i = m
    while curr_k > 0:
        prev_j = split[curr_k][curr_i]
        # segment from prev_j to curr_i-1 forms a route
        seg_cust = seq[prev_j:curr_i]  # list of customers
        route = [0] + seg_cust + [0]
        routes.append(route)
        curr_i = prev_j
        curr_k -= 1
    routes.reverse()  # because we built from last to first
    
    # Ensure exactly truck_count routes (truck_count could be larger than necessary, fill empty)
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    # Call report_best_vrp
    # report_best_vrp(routes)  # Uncomment if needed
    
    return routes