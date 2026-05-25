import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    
    # Nearest insertion construction
    def nearest_insertion():
        unvisited = set(range(n))
        start = np.random.randint(n)
        tour = [start]
        unvisited.remove(start)
        while unvisited:
            best_city = None
            best_cost = float('inf')
            best_pos = None
            for city in unvisited:
                for pos in range(len(tour) + 1):
                    # cost of inserting city at pos
                    if pos == 0:
                        cost = distance_matrix[city, tour[0]] + distance_matrix[tour[-1], city] - distance_matrix[tour[-1], tour[0]]
                    elif pos == len(tour):
                        cost = distance_matrix[tour[-1], city] + distance_matrix[city, tour[0]] - distance_matrix[tour[-1], tour[0]]
                    else:
                        before = tour[pos-1]
                        after = tour[pos]
                        cost = distance_matrix[before, city] + distance_matrix[city, after] - distance_matrix[before, after]
                    if cost < best_cost:
                        best_cost = cost
                        best_city = city
                        best_pos = pos
            tour.insert(best_pos, best_city)
            unvisited.remove(best_city)
        return np.array(tour, dtype=np.int32)
    
    def tour_distance(tour):
        s = 0
        for i in range(n):
            s += distance_matrix[tour[i], tour[(i+1)%n]]
        return s
    
    best_tour = nearest_insertion()
    best_dist = tour_distance(best_tour)
    report_best_tour(best_tour.copy())
    
    # 2-opt improvement with budget as number of passes
    tour = best_tour.copy()
    dist = best_dist
    iteration = 0
    while iteration < budget:
        improved = False
        for i in range(n-2):
            for j in range(i+2, n):
                a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                current = distance_matrix[a, b] + distance_matrix[c, d]
                new = distance_matrix[a, c] + distance_matrix[b, d]
                if new + 1e-12 < current:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    dist = dist - current + new
                    if dist < best_dist - 1e-12:
                        best_dist = dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour.copy())
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
        iteration += 1
    
    # If budget remains, try random perturbation and restart 2-opt
    while iteration < budget:
        # Perturb: swap two random non-adjacent cities
        i, j = np.random.choice(n, 2, replace=False)
        if abs(i - j) == 1 or (i == 0 and j == n-1) or (i == n-1 and j == 0):
            continue
        tour = best_tour.copy()
        tour[i], tour[j] = tour[j], tour[i]
        dist = tour_distance(tour)
        improved = True
        while improved and iteration < budget:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                    current = distance_matrix[a, b] + distance_matrix[c, d]
                    new = distance_matrix[a, c] + distance_matrix[b, d]
                    if new + 1e-12 < current:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        dist = dist - current + new
                        if dist < best_dist - 1e-12:
                            best_dist = dist
                            best_tour = tour.copy()
                            report_best_tour(best_tour.copy())
                        improved = True
                        break
                if improved:
                    break
            iteration += 1
    
    return best_tour