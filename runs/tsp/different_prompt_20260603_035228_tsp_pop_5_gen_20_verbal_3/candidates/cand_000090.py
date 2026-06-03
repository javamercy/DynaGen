import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n, dtype=int)
        report_best_tour(tour)
        return tour

    def total_dist(t):
        return distance_matrix[t[-1], t[0]] + np.sum(distance_matrix[t[:-1], t[1:]])

    def two_opt_first(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                    if distance_matrix[a, b] + distance_matrix[c, d] > distance_matrix[a, c] + distance_matrix[b, d] + 1e-12:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return tour

    def perturbation(tour):
        seg_len = np.random.randint(2, max(3, n//3 + 1))
        i = np.random.randint(0, n - seg_len)
        tour[i:i+seg_len] = tour[i:i+seg_len][::-1]
        return tour

    def greedy_start():
        start = np.random.randint(n)
        tour = [start]
        visited = [False] * n
        visited[start] = True
        for _ in range(n - 1):
            last = tour[-1]
            best = min((j for j in range(n) if not visited[j]), key=lambda j: distance_matrix[last, j])
            tour.append(best)
            visited[best] = True
        return np.array(tour, dtype=int)

    best_tour = None
    best_dist = float('inf')
    num_restarts = 5
    max_cycles = 20
    stall_limit = 5

    for _ in range(num_restarts):
        tour = greedy_start()
        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)

        no_improve = 0
        for _ in range(max_cycles):
            tour = two_opt_first(tour)
            cur_dist = total_dist(tour)
            if cur_dist < best_dist - 1e-12:
                best_dist = cur_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= stall_limit:
                tour = greedy_start()
                no_improve = 0
            else:
                tour = perturbation(tour)

    return best_tour