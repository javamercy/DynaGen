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
    
    # farthest pair as seed
    max_dist = -1
    start = 0
    second = 1
    for i in range(n):
        for j in range(i+1, n):
            d = distance_matrix[i, j]
            if d > max_dist:
                max_dist = d
                start, second = i, j
    
    current_tour = [start, second]
    visited = {start, second}
    dist = distance_matrix[start, second]
    budget_used = 0
    
    # nearest insertion: pick unvisited city with smallest insertion cost
    while budget_used < budget and len(current_tour) < n:
        best_city = None
        best_pos = None
        best_inc = np.inf
        for city in range(n):
            if city in visited:
                continue
            m = len(current_tour)
            for i in range(m):
                prev = current_tour[i]
                nxt = current_tour[(i+1)%m]
                inc = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
                if inc < best_inc:
                    best_inc = inc
                    best_city = city
                    best_pos = i
        current_tour.insert(best_pos+1, best_city)
        visited.add(best_city)
        dist += best_inc
        budget_used += 1
    
    if len(current_tour) == n:
        candidate = np.array(current_tour)
        if dist < best_dist:
            best_dist = dist
            best_tour = candidate.copy()
            report_best_tour(best_tour)
    
    # simple 2-opt: restart from beginning after each improvement
    remaining_budget = budget - budget_used
    while remaining_budget > 0:
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                if remaining_budget <= 0:
                    break
                a, b = best_tour[i], best_tour[(i+1)%n]
                c, d = best_tour[j], best_tour[(j+1)%n]
                delta = -distance_matrix[a,b] - distance_matrix[c,d] + distance_matrix[a,c] + distance_matrix[b,d]
                if delta < 0:
                    new_tour = best_tour.copy()
                    new_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                    best_tour = new_tour
                    best_dist += delta
                    report_best_tour(best_tour)
                    remaining_budget -= 1
                    improved = True
                    break
            if improved or remaining_budget <= 0:
                break
        if not improved:
            break
    
    return best_tour