import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = len(distance_matrix)

    # initial random tour
    tour = np.random.permutation(n)
    best_tour = tour.copy()
    best_dist = 0.0
    for i in range(n):
        best_dist += distance_matrix[tour[i], tour[(i+1)%n]]
    report_best_tour(best_tour)

    if budget <= 0 or n <= 2:
        return best_tour

    # Regret insertion construction
    # start with two random distinct cities
    start = np.random.randint(n)
    second = np.random.randint(n)
    while second == start:
        second = np.random.randint(n)
    current_tour = [start, second]
    visited = {start, second}
    dist = distance_matrix[start, second]

    while budget > 0 and len(current_tour) < n:
        best_regret = -np.inf
        best_city = None
        best_pos = None
        best_inc = None
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
                best_city = city
                best_pos = int(np.argmin(incs))
                best_inc = incs[best_pos]
        # Insert at best_pos+1
        current_tour.insert(best_pos+1, best_city)
        visited.add(best_city)
        dist += best_inc
        budget -= 1

    if len(current_tour) == n:
        candidate = np.array(current_tour)
        if dist < best_dist:
            best_dist = dist
            best_tour = candidate.copy()
            report_best_tour(best_tour)

    # 2-opt improvement
    while budget > 0:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                a, b = best_tour[i], best_tour[(i+1)%n]
                c, d = best_tour[j], best_tour[(j+1)%n]
                delta = -distance_matrix[a,b] - distance_matrix[c,d] + distance_matrix[a,c] + distance_matrix[b,d]
                if delta < 0:
                    new_tour = best_tour.copy()
                    new_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                    best_tour = new_tour
                    best_dist += delta
                    report_best_tour(best_tour)
                    budget -= 1
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best_tour