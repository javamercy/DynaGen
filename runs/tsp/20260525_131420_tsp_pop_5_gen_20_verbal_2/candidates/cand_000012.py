import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    # Regret-based insertion
    # Start with a random city
    start = np.random.randint(n)
    tour = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    # Initialize insertion costs for each unvisited city
    # For simplicity, we'll recompute costs each iteration (O(n^2) total)
    while unvisited:
        best_costs = {}  # city -> (cost, position, second_best_cost)
        for city in unvisited:
            min_cost = float('inf')
            min_pos = -1
            second_min_cost = float('inf')
            for i in range(len(tour)):
                a = tour[i]
                b = tour[(i+1) % len(tour)]
                cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                if cost < min_cost:
                    second_min_cost = min_cost
                    min_cost = cost
                    min_pos = i+1
                elif cost < second_min_cost:
                    second_min_cost = cost
            best_costs[city] = (min_cost, min_pos, second_min_cost)
        # Select city with maximum regret (difference between second best and best)
        # If only one unvisited, regret doesn't matter, choose that
        if len(unvisited) == 1:
            city = next(iter(unvisited))
        else:
            max_regret = -1
            city = None
            for c, (cost, pos, second) in best_costs.items():
                regret = second - cost
                if regret > max_regret:
                    max_regret = regret
                    city = c
        cost, pos, _ = best_costs[city]
        tour.insert(pos, city)
        unvisited.remove(city)
    tour = np.array(tour, dtype=np.int32)
    # Compute initial distance
    total = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    best_tour = tour.copy()
    best_dist = total
    report_best_tour(best_tour)
    # 2-opt improvement
    iteration = 0
    improved = True
    while improved and iteration < budget:
        improved = False
        for i in range(n-2):
            for j in range(i+2, n):
                iteration += 1
                if iteration > budget:
                    break
                a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < -1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    total += delta
                    if total < best_dist - 1e-12:
                        best_dist = total
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    improved = True
                    break
            if improved or iteration >= budget:
                break
    return best_tour