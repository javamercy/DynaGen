import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour)
        return tour

    np.random.seed(seed)

    # nearest neighbor initial tour
    start = np.random.randint(n)
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        next_city = min(unvisited, key=lambda x: distance_matrix[current, x])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    tour = np.array(tour, dtype=np.int32)
    report_best_tour(tour.copy())

    def compute_length(t):
        total = 0.0
        for i in range(n):
            total += distance_matrix[t[i], t[(i+1)%n]]
        return total

    best_tour = tour.copy()
    best_len = compute_length(best_tour)
    current_tour = tour.copy()
    current_len = best_len

    # temperature schedule
    T0 = 100.0
    T_end = 0.01
    alpha = (T_end / T0) ** (1.0 / max(budget, 1))
    iteration = 0

    while budget > 0:
        i = np.random.randint(n)
        j = np.random.randint(n)
        while abs(j - i) % n <= 1 or (j == i):
            j = np.random.randint(n)
        if i > j:
            i, j = j, i
        a = current_tour[i]
        b = current_tour[(i+1)%n]
        c = current_tour[j]
        d = current_tour[(j+1)%n]
        delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
        budget -= 1

        if delta < 0:
            # reverse segment i+1..j
            current_tour[i+1:j+1] = current_tour[i+1:j+1][::-1]
            current_len = compute_length(current_tour)
            if current_len < best_len:
                best_len = current_len
                best_tour = current_tour.copy()
                report_best_tour(best_tour.copy())
        else:
            T = T0 * (alpha ** iteration)
            iteration += 1
            if np.random.random() < np.exp(-delta / T):
                current_tour[i+1:j+1] = current_tour[i+1:j+1][::-1]
                current_len = compute_length(current_tour)
    return best_tour