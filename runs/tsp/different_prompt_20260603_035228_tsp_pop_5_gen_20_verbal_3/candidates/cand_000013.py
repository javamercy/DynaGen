import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=int)
        report_best_tour(tour.copy())
        return tour
    if n == 2:
        tour = np.array([0, 1], dtype=int)
        report_best_tour(tour.copy())
        return tour

    def compute_cost(tour):
        return distance_matrix[tour[-1], tour[0]] + np.sum(distance_matrix[tour[:-1], tour[1:]])

    # Farthest insertion construction
    tour = [0, 1]
    visited = {0, 1}
    for _ in range(2, n):
        best_node = None
        best_cost_increase = -np.inf  # maximize cost increase
        best_pos = None
        for node in range(n):
            if node not in visited:
                # Compute cost increase if inserted at the best position for this node
                # First find the best position for this node (minimizing cost increase)
                min_increase = np.inf
                best_pos_for_node = None
                for pos in range(len(tour) + 1):
                    if pos == 0:
                        increase = distance_matrix[node][tour[0]] + distance_matrix[tour[-1]][node] - distance_matrix[tour[-1]][tour[0]]
                    elif pos == len(tour):
                        increase = distance_matrix[tour[-1]][node] + distance_matrix[node][tour[0]] - distance_matrix[tour[-1]][tour[0]]
                    else:
                        increase = distance_matrix[tour[pos-1]][node] + distance_matrix[node][tour[pos]] - distance_matrix[tour[pos-1]][tour[pos]]
                    if increase < min_increase - 1e-10:
                        min_increase = increase
                        best_pos_for_node = pos
                # Use the minimum increase as the score for this node, but we want farthest, so we maximize that minimum increase
                if min_increase > best_cost_increase + 1e-10:
                    best_cost_increase = min_increase
                    best_node = node
                    best_pos = best_pos_for_node
        tour.insert(best_pos, best_node)
        visited.add(best_node)
    tour = np.array(tour, dtype=int)
    best_cost = compute_cost(tour)
    best_tour = tour.copy()
    report_best_tour(best_tour.copy())

    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                new_tour = tour.copy()
                new_tour[i+1:j+1] = new_tour[i+1:j+1][::-1]
                new_cost = compute_cost(new_tour)
                if new_cost < best_cost - 1e-10:
                    best_cost = new_cost
                    best_tour = new_tour.copy()
                    tour = new_tour.copy()
                    improved = True
                    report_best_tour(best_tour.copy())
                    break
            if improved:
                break
    return best_tour