import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=int)

    def compute_total_distance(tour):
        d = 0.0
        for i in range(n-1):
            d += distance_matrix[tour[i], tour[i+1]]
        d += distance_matrix[tour[-1], tour[0]]
        return d

    # Nearest neighbor initial tour
    visited = [0]
    current = 0
    available = set(range(1, n))
    while available:
        next_city = min(available, key=lambda x: distance_matrix[current, x])
        visited.append(next_city)
        available.remove(next_city)
        current = next_city
    best_tour = np.array(visited)
    best_dist = compute_total_distance(best_tour)
    report_best_tour(best_tour)

    # Farthest-insertion construction
    start = 0
    tour = [start]
    remaining = set(range(1, n))
    # first city farthest from start
    farthest = max(remaining, key=lambda x: distance_matrix[start, x])
    tour.append(farthest)
    remaining.remove(farthest)
    while remaining:
        # for each remaining city, compute minimum distance to current tour
        min_dists = {}
        for city in remaining:
            min_dist = min(distance_matrix[city, node] for node in tour)
            min_dists[city] = min_dist
        chosen = max(min_dists, key=lambda x: min_dists[x])
        # find best insertion position
        best_cost = float('inf')
        best_idx = -1
        for i in range(len(tour)):
            prev = tour[i]
            nxt = tour[(i+1) % len(tour)]
            cost = distance_matrix[prev, chosen] + distance_matrix[chosen, nxt] - distance_matrix[prev, nxt]
            if cost < best_cost:
                best_cost = cost
                best_idx = i+1
        tour.insert(best_idx, chosen)
        remaining.remove(chosen)
    tour_farthest = np.array(tour)
    dist_farthest = compute_total_distance(tour_farthest)
    if dist_farthest < best_dist - 1e-12:
        best_dist = dist_farthest
        best_tour = tour_farthest.copy()
        report_best_tour(best_tour)

    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+2, n):
                new_tour = np.concatenate([best_tour[:i+1], best_tour[j:i:-1], best_tour[j+1:]])
                new_dist = compute_total_distance(new_tour)
                if new_dist < best_dist - 1e-12:
                    best_dist = new_dist
                    best_tour = new_tour
                    improved = True
                    report_best_tour(best_tour)
    return best_tour