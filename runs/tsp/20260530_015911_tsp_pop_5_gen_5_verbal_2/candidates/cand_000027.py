import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    best_tour = None
    best_dist = np.inf

    def two_opt(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n - 2):
                for j in range(i + 2, n):
                    a, b = tour[i], tour[(i+1) % n]
                    c, d = tour[j], tour[(j+1) % n]
                    delta = (distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d])
                    if delta < -1e-12:
                        tour[i+1:j+1] = np.flip(tour[i+1:j+1])
                        improved = True
                        break
                if improved:
                    break
        return tour

    def double_bridge(tour):
        arr = tour.copy()
        i, j, k, l = np.random.randint(0, n, 4)
        a, b, c, d = sorted([i, j, k, l])
        seg1 = arr[a:b]
        seg2 = arr[b:c]
        seg3 = arr[c:d]
        seg4 = np.concatenate([arr[d:], arr[:a]])
        return np.concatenate([seg1, seg4, seg3, seg2])

    for restart in range(6):
        # Random-start nearest neighbor
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            last = tour[-1]
            nearest = min(unvisited, key=lambda x: distance_matrix[last, x])
            tour.append(nearest)
            unvisited.remove(nearest)
        tour = np.array(tour, dtype=np.int32)
        tour = two_opt(tour)
        current_dist = np.sum(distance_matrix[tour[:-1], tour[1:]]) + distance_matrix[tour[-1], tour[0]]
        if current_dist < best_dist - 1e-12:
            best_dist = current_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)

        # Simulated annealing schedule
        T0 = 0.1 * np.max(distance_matrix)
        T = T0
        num_iters = 25

        for iteration in range(num_iters):
            perturbed = double_bridge(tour)
            new_tour = two_opt(perturbed)
            new_dist = np.sum(distance_matrix[new_tour[:-1], new_tour[1:]]) + distance_matrix[new_tour[-1], new_tour[0]]
            delta = new_dist - current_dist
            if delta < -1e-12 or np.random.random() < np.exp(-delta / T):
                tour = new_tour
                current_dist = new_dist
                if current_dist < best_dist - 1e-12:
                    best_dist = current_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
            T = T0 * (1 - (iteration + 1) / num_iters)

    # Final 2-opt on best tour
    best_tour = two_opt(best_tour.copy())
    best_dist = np.sum(distance_matrix[best_tour[:-1], best_tour[1:]]) + distance_matrix[best_tour[-1], best_tour[0]]
    report_best_tour(best_tour)
    return best_tour