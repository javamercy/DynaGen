import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    
    k = min(n - 1, 20)
    nn = [np.argsort(distance_matrix[i])[1:k+1].tolist() for i in range(n)]
    
    def build_pos(tour):
        pos = np.empty(n, dtype=np.int32)
        for idx, city in enumerate(tour):
            pos[city] = idx
        return pos
    
    def tour_dist(tour):
        return sum(distance_matrix[tour[i], tour[(i+1) % n]] for i in range(n))
    
    def regret_insertion():
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            best = []
            for city in unvisited:
                min_cost = float('inf')
                second_min = float('inf')
                best_pos = -1
                for i in range(len(tour)):
                    a = tour[i]
                    b = tour[(i+1) % len(tour)]
                    cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                    if cost < min_cost:
                        second_min = min_cost
                        min_cost = cost
                        best_pos = i + 1
                    elif cost < second_min:
                        second_min = cost
                regret = second_min - min_cost
                best.append((city, min_cost, best_pos, regret))
            max_regret = max(c[3] for c in best)
            candidates = [c for c in best if c[3] == max_regret]
            city, _, pos, _ = candidates[np.random.randint(len(candidates))]
            tour.insert(pos, city)
            unvisited.remove(city)
        return np.array(tour, dtype=np.int32)
    
    tour = regret_insertion()
    pos = build_pos(tour)
    best_tour = tour.copy()
    best_dist = tour_dist(tour)
    report_best_tour(best_tour)
    
    local_dist = best_dist
    last_improvement = 0
    iteration = 0
    restart_threshold = max(10, budget // 10)
    improved = True
    
    while iteration < budget and improved:
        improved = False
        for i in range(n):
            a = tour[i]
            b = tour[(i+1) % n]
            for c in nn[b]:
                j = pos[c]
                if j <= i + 1 or j >= n - 1:
                    continue
                d = tour[(j+1) % n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                iteration += 1
                if iteration >= budget:
                    break
                if delta < -1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    for idx in range(i+1, j+1):
                        pos[tour[idx]] = idx
                    local_dist += delta
                    if local_dist < best_dist - 1e-12:
                        best_dist = local_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    improved = True
                    last_improvement = iteration
                    break
            if improved:
                break
            if iteration >= budget:
                break
        if not improved and iteration < budget:
            if iteration - last_improvement > restart_threshold:
                tour = regret_insertion()
                pos = build_pos(tour)
                local_dist = tour_dist(tour)
                if local_dist < best_dist - 1e-12:
                    best_dist = local_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                last_improvement = iteration
                improved = True
            else:
                break
    return best_tour