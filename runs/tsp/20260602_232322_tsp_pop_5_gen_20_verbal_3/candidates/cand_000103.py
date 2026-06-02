import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n)
    # farthest insertion initialization
    start = np.random.randint(n)
    tour = [start]
    visited = [False] * n
    visited[start] = True
    while len(tour) < n:
        # find farthest unvisited node
        max_min_dist = -1
        farthest_node = -1
        for node in range(n):
            if not visited[node]:
                min_dist = min(distance_matrix[node][t] for t in tour)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    farthest_node = node
        # find best insertion position
        best_cost = float('inf')
        best_pos = 0
        for pos in range(len(tour)):
            prev = tour[pos-1] if pos > 0 else tour[-1]
            nxt = tour[pos]
            cost = distance_matrix[prev][farthest_node] + distance_matrix[farthest_node][nxt] - distance_matrix[prev][nxt]
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        tour.insert(best_pos, farthest_node)
        visited[farthest_node] = True
    tour = np.array(tour)
    best_tour = tour.copy()
    best_dist = sum(distance_matrix[tour[i-1], tour[i]] for i in range(n))
    report_best_tour(best_tour)
    
    # simulated annealing
    T = 1.0
    T_min = 1e-4
    alpha = 0.995
    while T > T_min:
        i, j = np.random.choice(n, 2, replace=False)
        if i > j:
            i, j = j, i
        if (j - i) % n == 1 or (i - j) % n == 1:
            continue
        a, b = tour[i], tour[(i+1)%n]
        c, d = tour[j], tour[(j+1)%n]
        delta = distance_matrix[a,c] + distance_matrix[b,d] - (distance_matrix[a,b] + distance_matrix[c,d])
        if delta < 0 or np.random.rand() < np.exp(-delta/T):
            segment = tour[i+1:j+1]
            tour[i+1:j+1] = segment[::-1]
            new_dist = sum(distance_matrix[tour[k-1], tour[k]] for k in range(n))
            if new_dist < best_dist:
                best_dist = new_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
        T *= alpha
    return best_tour