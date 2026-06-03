import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.array([0]) if n == 1 else np.array([0, 1])
        report_best_tour(tour)
        return tour

    def total_dist(t):
        idx = np.array(t)
        return distance_matrix[idx[-1], idx[0]] + np.sum(distance_matrix[idx[:-1], idx[1:]])

    def two_opt_first(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    delta = (distance_matrix[tour[i], tour[i+1]] +
                             distance_matrix[tour[j], tour[(j+1)%n]] -
                             distance_matrix[tour[i], tour[j]] -
                             distance_matrix[tour[i+1], tour[(j+1)%n]])
                    if delta > 1e-12:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
                        break
                if improved:
                    break
        return tour

    def farthest_insertion(start):
        tour = [start]
        visited = {start}
        while len(tour) < n:
            farthest_node = None
            farthest_dist = -1
            for v in range(n):
                if v not in visited:
                    min_dist = min(distance_matrix[v, tour_node] for tour_node in tour)
                    if min_dist > farthest_dist:
                        farthest_dist = min_dist
                        farthest_node = v
            best_gain = float('inf')
            best_pos = 0
            for pos in range(len(tour)):
                next_pos = (pos + 1) % len(tour)
                gain = (distance_matrix[tour[pos], farthest_node] +
                        distance_matrix[farthest_node, tour[next_pos]] -
                        distance_matrix[tour[pos], tour[next_pos]])
                if gain < best_gain:
                    best_gain = gain
                    best_pos = pos + 1
            tour.insert(best_pos, farthest_node)
            visited.add(farthest_node)
        return tour

    best_tour = None
    best_dist = float('inf')
    max_restarts = 5
    for restart in range(max_restarts):
        start_node = restart % n
        tour = farthest_insertion(start_node)
        tour = two_opt_first(tour)
        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = np.array(tour)
            report_best_tour(best_tour)
    return best_tour