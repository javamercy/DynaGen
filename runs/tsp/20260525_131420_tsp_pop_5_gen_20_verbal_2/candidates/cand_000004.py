import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour)
        return tour

    # Nearest neighbor construction
    tour = np.zeros(n, dtype=np.int32)
    start = random.randint(0, n-1)
    tour[0] = start
    visited = {start}
    for i in range(1, n):
        last = tour[i-1]
        best = -1
        best_dist = np.inf
        for j in range(n):
            if j not in visited:
                d = distance_matrix[last, j]
                if d < best_dist:
                    best_dist = d
                    best = j
        tour[i] = best
        visited.add(best)

    # Ensure numpy array and int type
    tour = tour.astype(np.int32)
    best_tour = tour.copy()
    best_cost = tour_cost(tour, distance_matrix)
    report_best_tour(best_tour)

    # Lin-Kernighan improvement
    def tour_cost(t, dm):
        cost = dm[t[-1], t[0]]
        for i in range(len(t)-1):
            cost += dm[t[i], t[i+1]]
        return cost

    def improve(t):
        n = len(t)
        improved = False
        for i in range(n):
            t1 = t[i]
            t2 = t[(i+1)%n]
            for j in range(n):
                if j == i or j == (i+1)%n or j == (i-1)%n:
                    continue
                t3 = t[j]
                t4 = t[(j+1)%n]
                # Consider reversal of segment between t2 and t4
                cost_diff = (distance_matrix[t1, t4] + distance_matrix[t2, t3]) - (distance_matrix[t1, t2] + distance_matrix[t3, t4])
                if cost_diff < -1e-12:
                    # Reverse segment
                    if i < j:
                        t[i+1:j+1] = t[i+1:j+1][::-1]
                    else:
                        # wrap around
                        seg = np.concatenate((t[i+1:], t[:j+1]))
                        seg = seg[::-1]
                        t[i+1:] = seg[:n-i-1]
                        t[:j+1] = seg[n-i-1:]
                    improved = True
        return improved

    # Variable depth: try sequence of 2-opt moves (simplified LK)
    # Since budget is limited, we do a simple 2-opt iterative improvement
    # This is not full LK but works as a compact improver.
    while budget > 0:
        new_tour = best_tour.copy()
        improved = False
        for _ in range(budget):
            if improve(new_tour):
                improved = True
                cost_new = tour_cost(new_tour, distance_matrix)
                if cost_new < best_cost - 1e-12:
                    best_tour = new_tour.copy()
                    best_cost = cost_new
                    report_best_tour(best_tour)
                break
            else:
                budget -= 1
                if budget <= 0:
                    break
        if not improved:
            break
        budget -= 1

    return best_tour