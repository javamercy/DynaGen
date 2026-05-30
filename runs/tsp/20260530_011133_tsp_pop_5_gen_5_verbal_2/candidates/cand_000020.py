import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=int)

    def total_distance(tour):
        d = 0.0
        for i in range(n - 1):
            d += distance_matrix[tour[i], tour[i + 1]]
        d += distance_matrix[tour[-1], tour[0]]
        return d

    def random_greedy_tour():
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        while unvisited:
            candidates = sorted(unvisited, key=lambda x: distance_matrix[current, x])
            k = max(1, int(np.sqrt(len(unvisited))))
            # pick randomly among top k
            idx = np.random.randint(min(k, len(candidates)))
            next_city = candidates[idx]
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        return np.array(tour, dtype=int)

    def two_opt(tour):
        improved = True
        while improved:
            improved = False
            best_dist = total_distance(tour)
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    new_tour = np.concatenate([tour[:i], tour[j:i-1:-1], tour[j+1:]])
                    new_dist = total_distance(new_tour)
                    if new_dist < best_dist - 1e-12:
                        tour = new_tour
                        best_dist = new_dist
                        improved = True
                        report_best_tour(tour)
            if improved:
                continue
        return tour, best_dist

    def double_bridge(tour):
        # Ensure at least 4 cities to perform double bridge
        if n < 4:
            return tour
        while True:
            p = np.random.randint(1, n - 3)
            q = np.random.randint(p + 1, n - 2)
            r = np.random.randint(q + 1, n - 1)
            s = np.random.randint(r + 1, n)
            # Ensure segments are non-empty
            if p >= 1 and q > p and r > q and s > r and s < n:
                break
        # Reorder: A(0..p), B(p+1..q), C(q+1..r), D(r+1..s), E(s+1..n-1)
        # New tour: A, D, C, B, E
        new_tour = np.concatenate([tour[:p+1], tour[r+1:s+1], tour[q+1:r+1], tour[p+1:q+1], tour[s+1:]])
        return new_tour

    # Initial solution
    best_tour, best_dist = two_opt(random_greedy_tour())
    report_best_tour(best_tour)

    # Iterated local search
    max_iter = 20 if n <= 50 else 10
    no_improve = 0
    for _ in range(max_iter):
        # Perturbation
        perturbed = double_bridge(best_tour.copy())
        # Improve with 2-opt
        new_tour, new_dist = two_opt(perturbed)
        if new_dist < best_dist - 1e-12:
            best_tour = new_tour
            best_dist = new_dist
            report_best_tour(best_tour)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 5:
                # Restart with new randomized tour
                new_tour, new_dist = two_opt(random_greedy_tour())
                if new_dist < best_dist - 1e-12:
                    best_tour = new_tour
                    best_dist = new_dist
                    report_best_tour(best_tour)
                no_improve = 0

    return best_tour