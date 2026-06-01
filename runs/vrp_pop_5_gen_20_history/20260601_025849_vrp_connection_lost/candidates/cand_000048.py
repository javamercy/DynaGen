import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    # Step 1: Build giant TSP tour using nearest neighbor insertion
    # Start with depot, then repeatedly add the nearest unvisited customer to the end
    tour = [0]
    unvisited = set(customers)
    while unvisited:
        last = tour[-1]
        # find nearest unvisited customer, tie by smallest index
        best_cust = min(unvisited, key=lambda c: (distance_matrix[last, c], c))
        tour.append(best_cust)
        unvisited.remove(best_cust)
    tour.append(0)  # close the tour
    
    # Remove the trailing 0 for sequence of customers
    seq = tour[1:-1]  # list of customer indices in order
    m = len(seq)
    
    # Step 2: Precompute segment distances
    segment_dist = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i, m):
            d = distance_matrix[0, seq[i]]
            for k in range(i, j):
                d += distance_matrix[seq[k], seq[k+1]]
            d += distance_matrix[seq[j], 0]
            segment_dist[i][j] = d
    
    # Step 3: DP for minimum max distance partition
    INF = float('inf')
    dp = [[INF] * (m+1) for _ in range(truck_count+1)]
    split = [[-1] * (m+1) for _ in range(truck_count+1)]
    
    # Base: k=1
    for i in range(1, m+1):
        dp[1][i] = segment_dist[0][i-1]
        split[1][i] = 0
    
    # Fill dp for k>=2
    for k in range(2, truck_count+1):
        for i in range(k, m+1):
            best_val = INF
            best_j = -1
            for j in range(k-1, i):
                val = max(dp[k-1][j], segment_dist[j][i-1])
                if val < best_val:
                    best_val = val
                    best_j = j
            dp[k][i] = best_val
            split[k][i] = best_j
    
    # Reconstruct routes
    routes = []
    curr_k = truck_count
    curr_i = m
    while curr_k > 0:
        prev_j = split[curr_k][curr_i]
        seg_cust = seq[prev_j:curr_i]
        route = [0] + seg_cust + [0]
        routes.append(route)
        curr_i = prev_j
        curr_k -= 1
    routes.reverse()
    
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    # report_best_vrp(routes)
    return routes