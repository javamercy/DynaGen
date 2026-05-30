import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # farthest insertion construction
    start = 0
    end = np.argmax(distance_matrix[start])
    tour = [start, end]
    in_tour = {start, end}
    while len(tour) < n:
        best_node = -1
        best_dist = -1.0
        for node in range(n):
            if node in in_tour:
                continue
            min_dist = min(distance_matrix[node][t] for t in tour)
            if min_dist > best_dist:
                best_dist = min_dist
                best_node = node
        best_pos = -1
        best_increase = float('inf')
        for i in range(len(tour)):
            a = tour[i]
            b = tour[(i+1)%len(tour)]
            increase = distance_matrix[a][best_node] + distance_matrix[best_node][b] - distance_matrix[a][b]
            if increase < best_increase:
                best_increase = increase
                best_pos = i+1
        tour.insert(best_pos, best_node)
        in_tour.add(best_node)
    best_tour = tour.copy()
    best_cost = total_distance(best_tour, distance_matrix)
    report_best_tour(np.array(best_tour))
    # 2-opt local search (best improvement)
    improved = True
    while improved:
        improved = False
        best_delta = 0
        best_i = best_j = -1
        for i in range(n):
            for j in range(i+2, n):
                if i == 0 and j == n-1:
                    continue
                # delta = new cost - old cost for reversing segment (i+1..j)
                a = tour[i]
                b = tour[(i+1)%n]
                c = tour[j]
                d = tour[(j+1)%n]
                delta = (distance_matrix[a][c] + distance_matrix[b][d]) - (distance_matrix[a][b] + distance_matrix[c][d])
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        if best_delta < 0:
            i, j = best_i, best_j
            tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
            new_cost = total_distance(tour, distance_matrix)
            if new_cost < best_cost:
                best_cost = new_cost
                best_tour = tour.copy()
                report_best_tour(np.array(best_tour))
            improved = True
    return np.array(best_tour)

def total_distance(tour, dist):
    return sum(dist[tour[i]][tour[(i+1)%len(tour)]] for i in range(len(tour)))