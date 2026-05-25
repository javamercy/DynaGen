import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(distance_matrix)
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    # --- Randomized Cheapest Insertion Construction ---
    start = rng.integers(n)
    tour = [start]
    unvisited = set(range(n)) - {start}
    k = max(3, int(np.sqrt(n)))
    while unvisited:
        best_inc = {}
        for city in unvisited:
            min_inc = np.inf
            best_pos = -1
            m = len(tour)
            for i in range(m):
                prev = tour[i]
                nxt = tour[(i + 1) % m]
                inc = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
                if inc < min_inc:
                    min_inc = inc
                    best_pos = i + 1
            best_inc[city] = (min_inc, best_pos)
        sorted_cities = sorted(unvisited, key=lambda c: best_inc[c][0])
        rcl_size = min(k, len(sorted_cities))
        chosen = sorted_cities[rng.integers(rcl_size)]
        pos = best_inc[chosen][1]
        tour.insert(pos, chosen)
        unvisited.remove(chosen)
    tour = np.array(tour, dtype=np.int64)
    
    def tour_cost(t):
        cost = distance_matrix[t[-1], t[0]]
        for i in range(n - 1):
            cost += distance_matrix[t[i], t[i + 1]]
        return cost
    
    best_tour = tour.copy()
    best_cost = tour_cost(tour)
    report_best_tour(best_tour)
    
    # --- 2-opt Improvement (first-improvement, systematic scanning) ---
    cost = best_cost
    ops = 0
    improved = True
    while ops < budget and improved:
        improved = False
        for i in range(n - 1):
            if ops >= budget:
                break
            for j in range(i + 2, n):
                if ops >= budget:
                    break
                a, b = tour[i], tour[(i + 1) % n]
                c, d = tour[j], tour[(j + 1) % n]
                old = distance_matrix[a, b] + distance_matrix[c, d]
                new = distance_matrix[a, c] + distance_matrix[b, d]
                if new < old - 1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    cost += new - old
                    ops += 1
                    improved = True
                    if cost < best_cost:
                        best_cost = cost
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break  # continue scanning with next i
            # if just broke inner loop, continue outer loop
        # end of pass
    return best_tour