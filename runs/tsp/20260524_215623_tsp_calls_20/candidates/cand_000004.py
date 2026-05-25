import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    rng = np.random.default_rng(seed)
    # initial random tour
    tour = rng.permutation(n)
    best_tour = tour.copy()
    best_len = 0.0
    for i in range(n):
        best_len += distance_matrix[tour[i], tour[(i+1)%n]]
    report_best_tour(best_tour)
    if n <= 2 or budget <= 0:
        return best_tour
    # Regret insertion construction
    start = rng.integers(n)
    second = rng.integers(n)
    while second == start:
        second = rng.integers(n)
    current_tour = [start, second]
    visited = {start, second}
    current_len = distance_matrix[start, second]
    while budget > 0 and len(current_tour) < n:
        best_regret = -np.inf
        best_cities = []
        best_positions = []
        best_incs = []
        for city in range(n):
            if city in visited:
                continue
            m = len(current_tour)
            incs = np.zeros(m)
            for i in range(m):
                prev = current_tour[i]
                nxt = current_tour[(i+1)%m]
                incs[i] = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
            sorted_incs = np.sort(incs)
            regret = sorted_incs[1] - sorted_incs[0]
            if regret > best_regret:
                best_regret = regret
                best_cities = [city]
                best_positions = [int(np.argmin(incs))]
                best_incs = [incs[best_positions[0]]]
            elif abs(regret - best_regret) < 1e-12:
                best_cities.append(city)
                best_positions.append(int(np.argmin(incs)))
                best_incs.append(incs[best_positions[-1]])
        # random tie-breaking among best regret cities
        idx = rng.integers(len(best_cities))
        city = best_cities[idx]
        pos = best_positions[idx]
        inc = best_incs[idx]
        current_tour.insert(pos+1, city)
        visited.add(city)
        current_len += inc
        budget -= 1
        if current_len < best_len:
            best_len = current_len
            best_tour = np.array(current_tour)
            report_best_tour(best_tour)
    # 2-opt improvement
    while budget > 0:
        improved = False
        for _ in range(min(20, budget)):
            if budget <= 0:
                break
            i = rng.integers(n)
            j = rng.integers(n)
            if j == i or (j+1)%n == i or (i+1)%n == j:
                continue
            if j < i:
                i, j = j, i
            a = best_tour[i]
            b = best_tour[(i+1)%n]
            c = best_tour[j]
            d = best_tour[(j+1)%n]
            delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
            if delta < -1e-12:
                new_tour = best_tour.copy()
                new_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                best_tour = new_tour
                best_len += delta
                report_best_tour(best_tour)
                budget -= 1
                improved = True
                break
        if not improved:
            break
    return best_tour