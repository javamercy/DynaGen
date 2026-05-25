import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    
    k = min(n-1, 30)
    nn = [np.argsort(distance_matrix[i])[1:k+1] for i in range(n)]
    
    def compute_distance(tour):
        d = 0.0
        for i in range(n):
            d += distance_matrix[tour[i], tour[(i+1)%n]]
        return d
    
    def regret_insertion(start):
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            best = []
            for city in unvisited:
                best_cost = float('inf')
                best_pos = -1
                second = float('inf')
                for i in range(len(tour)):
                    a = tour[i]
                    b = tour[(i+1) % len(tour)]
                    cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                    if cost < best_cost:
                        second = best_cost
                        best_cost = cost
                        best_pos = i+1
                    elif cost < second:
                        second = cost
                regret = second - best_cost
                best.append((city, best_cost, best_pos, regret))
            max_r = max(b[3] for b in best)
            cand = [b for b in best if b[3] == max_r]
            city, _, pos, _ = cand[np.random.randint(len(cand))]
            tour.insert(pos, city)
            unvisited.remove(city)
        return np.array(tour, dtype=np.int32)
    
    # Multiple restarts for construction
    best_tour = None
    best_dist = float('inf')
    for _ in range(10):
        start = np.random.randint(n)
        tour = regret_insertion(start)
        dist = compute_distance(tour)
        if dist < best_dist:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
    
    if budget <= n:  # No budget for search
        return best_tour
    
    tour = best_tour.copy()
    curr_dist = best_dist
    pos = np.empty(n, dtype=int)
    for idx, city in enumerate(tour):
        pos[city] = idx
    
    iteration = 0
    last_improvement = 0
    stagnation_threshold = max(10, budget // 20)
    
    while iteration < budget:
        # Steepest-ascent 2-opt
        best_move = None
        best_delta = 0.0
        for i in range(n-2):
            a = tour[i]
            b = tour[i+1]
            for c in nn[b]:
                j = pos[c]
                if j <= i+1 or j >= n-1:
                    continue
                d = tour[(j+1) % n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_move = (i, j)
                iteration += 1
                if iteration >= budget:
                    break
            if iteration >= budget:
                break
        if iteration >= budget:
            break
        
        if best_move is not None and best_delta < -1e-12:
            i, j = best_move
            tour[i+1:j+1] = tour[i+1:j+1][::-1]
            for idx in range(i+1, j+1):
                pos[tour[idx]] = idx
            curr_dist += best_delta
            if curr_dist < best_dist - 1e-12:
                best_dist = curr_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
            last_improvement = iteration
        else:
            # Check stagnation
            if iteration - last_improvement > stagnation_threshold:
                # Random 2-opt swap kick
                i = np.random.randint(0, n-2)
                j = np.random.randint(i+2, n-1)
                delta = distance_matrix[tour[i], tour[j]] + distance_matrix[tour[i+1], tour[(j+1)%n]] - distance_matrix[tour[i], tour[i+1]] - distance_matrix[tour[j], tour[(j+1)%n]]
                tour[i+1:j+1] = tour[i+1:j+1][::-1]
                for idx in range(i+1, j+1):
                    pos[tour[idx]] = idx
                curr_dist += delta
                if curr_dist < best_dist - 1e-12:
                    best_dist = curr_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                last_improvement = iteration
                iteration += 1
                continue
        iteration += 1  # Account for move application or no move
    
    return best_tour