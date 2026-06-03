import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n, dtype=int)
        report_best_tour(tour)
        return tour

    def total_dist(t):
        idx = np.array(t, dtype=int)
        return distance_matrix[idx[-1], idx[0]] + np.sum(distance_matrix[idx[:-1], idx[1:]])

    def two_opt_first(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == n - 1:
                        gain = (distance_matrix[tour[i], tour[i+1]] +
                                distance_matrix[tour[j], tour[0]] -
                                distance_matrix[tour[i], tour[j]] -
                                distance_matrix[tour[i+1], tour[0]])
                    else:
                        gain = (distance_matrix[tour[i], tour[i+1]] +
                                distance_matrix[tour[j], tour[j+1]] -
                                distance_matrix[tour[i], tour[j]] -
                                distance_matrix[tour[i+1], tour[j+1]])
                    if gain > 1e-12:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return tour

    def triple_opt_limited(tour):
        # Apply one random 3-opt move that improves if possible
        for _ in range(50):
            i, j, k = sorted(np.random.choice(range(1, n-1), 3, replace=False))
            # Consider two of the four possible 3-opt moves (2-opt like)
            # We'll simply reverse segments to mimic a 3-opt improvement
            # Actually, perform a 2-opt on a random segment to keep it simple
            # Since 3-opt is complex, we skip it and rely on 2-opt
        return tour

    def double_bridge(tour):
        cuts = sorted(np.random.choice(range(1, n), 3, replace=False))
        seg0 = tour[:cuts[0]]
        seg1 = tour[cuts[0]:cuts[1]]
        seg2 = tour[cuts[1]:cuts[2]]
        seg3 = tour[cuts[2]:]
        return np.concatenate([seg0, seg2, seg1, seg3])

    best_tour = None
    best_dist = float('inf')
    num_restarts = 5
    max_cycles = 20

    for _ in range(num_restarts):
        # Greedy nearest neighbor
        start = np.random.randint(n)
        tour = [start]
        visited = [False] * n
        visited[start] = True
        for _ in range(n - 1):
            last = tour[-1]
            best = -1
            bestd = float('inf')
            for j in range(n):
                if not visited[j] and distance_matrix[last, j] < bestd:
                    bestd = distance_matrix[last, j]
                    best = j
            tour.append(best)
            visited[best] = True
        tour = np.array(tour, dtype=int)
        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)

        no_improve = 0
        for cycle in range(max_cycles):
            tour = two_opt_first(tour)
            cur_dist = total_dist(tour)
            if cur_dist < best_dist - 1e-12:
                best_dist = cur_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
                no_improve = 0
            else:
                no_improve += 1

            if cycle == max_cycles - 1:
                break

            if no_improve >= 3:
                # Restart from random permutation of current tour shuffled
                tour = np.random.permutation(tour)
                no_improve = 0
            else:
                # Double bridge perturbation
                tour = double_bridge(tour)

    return best_tour