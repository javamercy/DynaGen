import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(distance_matrix)
    # Probabilistic nearest neighbor construction
    unvisited = set(range(n))
    start = rng.integers(n)
    tour = [start]
    unvisited.remove(start)
    current = start
    while unvisited:
        # compute distances to unvisited
        dists = np.array([distance_matrix[current, u] for u in unvisited])
        # avoid division by zero
        inv_dist = 1.0 / (dists + 1e-10)
        probs = inv_dist / inv_dist.sum()
        # select next city according to probabilities
        idx = rng.choice(len(unvisited), p=probs)
        next_city = list(unvisited)[idx]
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    tour = np.array(tour, dtype=int)
    
    def tour_length(t):
        return distance_matrix[t[-1], t[0]] + np.sum(distance_matrix[t[:-1], t[1:]])
    best_tour = tour.copy()
    best_len = tour_length(tour)
    report_best_tour(best_tour)
    # 2-opt improvement
    improved = True
    while budget > 0 and improved:
        improved = False
        for i in range(n - 2):
            if budget <= 0:
                break
            for j in range(i + 2, n):
                if budget <= 0:
                    break
                a, b = tour[i], tour[i+1]
                c, d = tour[j], tour[(j+1)%n]
                old = distance_matrix[a,b] + distance_matrix[c,d]
                new = distance_matrix[a,c] + distance_matrix[b,d]
                if new < old:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    new_len = best_len - old + new
                    if new_len < best_len:
                        best_len = new_len
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                budget -= 1
    return best_tour