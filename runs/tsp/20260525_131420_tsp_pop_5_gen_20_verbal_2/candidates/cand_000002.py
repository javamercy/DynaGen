import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    rng = np.random.default_rng(seed)
    # Cheapest insertion heuristic
    tour = [rng.integers(n)]
    unvisited = set(range(n)) - {tour[0]}
    for _ in range(1, n):
        best_inc = np.inf
        best_city = None
        best_pos = None
        for city in unvisited:
            for pos in range(len(tour) + 1):
                if pos == 0:
                    inc = distance_matrix[city, tour[0]] + distance_matrix[tour[-1], city] - distance_matrix[tour[-1], tour[0]]
                elif pos == len(tour):
                    inc = distance_matrix[tour[-1], city] + distance_matrix[city, tour[0]] - distance_matrix[tour[-1], tour[0]]
                else:
                    inc = distance_matrix[tour[pos-1], city] + distance_matrix[city, tour[pos]] - distance_matrix[tour[pos-1], tour[pos]]
                if inc < best_inc:
                    best_inc = inc
                    best_city = city
                    best_pos = pos
        tour.insert(best_pos, best_city)
        unvisited.remove(best_city)
    tour = np.array(tour, dtype=np.int64)
    best_tour = tour.copy()
    best_cost = compute_cost(tour, distance_matrix)
    # 2-opt improvement
    ops = 0
    improved = True
    while improved and ops < budget:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                ops += 1
                if ops > budget:
                    break
                # reverse segment from i+1 to j (inclusive)
                new_tour = tour.copy()
                new_tour[i+1:j+1] = tour[j:i:-1] if i+1 < j else new_tour
                new_cost = compute_cost(new_tour, distance_matrix)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_tour = new_tour.copy()
                    tour = new_tour
                    improved = True
                    report_best_tour(best_tour)
                    break
            if ops > budget or improved:
                break
        # If no improvement, try different starting order? Use random restart?
        # To avoid infinite loop, we just break after full pass without improvement.
        if not improved and ops < budget:
            # random perturbation: reverse a random segment
            i = rng.integers(0, n-1)
            j = rng.integers(i+1, n)
            tour[i+1:j+1] = tour[j:i:-1]
            ops += 1
            improved = True  # allow another pass
    return best_tour

def compute_cost(tour, dist):
    return np.sum(dist[tour, np.roll(tour, -1)])