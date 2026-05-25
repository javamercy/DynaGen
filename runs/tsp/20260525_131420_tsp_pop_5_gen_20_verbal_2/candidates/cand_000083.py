import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    k = min(n - 1, 50)
    nn_list = [np.argsort(distance_matrix[i])[1:k+1] for i in range(n)]
    
    def build_pos(tour):
        pos = np.empty(n, dtype=np.int32)
        for idx, city in enumerate(tour):
            pos[city] = idx
        return pos
    
    def regret_insertion(start=None):
        if start is None:
            start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            candidates = []
            for city in unvisited:
                best_cost = float('inf')
                best_pos = -1
                second_best = float('inf')
                for i in range(len(tour)):
                    a = tour[i]
                    b = tour[(i + 1) % len(tour)]
                    cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                    if cost < best_cost:
                        second_best = best_cost
                        best_cost = cost
                        best_pos = i + 1
                    elif cost < second_best:
                        second_best = cost
                regret = second_best - best_cost
                candidates.append((city, best_cost, best_pos, regret))
            max_regret = max(c[3] for c in candidates)
            chosen = [c for c in candidates if c[3] == max_regret]
            city, _, pos, _ = chosen[np.random.randint(len(chosen))]
            tour.insert(pos, city)
            unvisited.remove(city)
        return np.array(tour, dtype=np.int32)
    
    def tour_distance(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))
    
    tour = regret_insertion()
    pos = build_pos(tour)
    current_dist = tour_distance(tour)
    best_tour = tour.copy()
    best_dist = current_dist
    report_best_tour(best_tour)
    
    iteration = 0
    last_improvement = 0
    restart_threshold = budget // 8
    
    while iteration < budget:
        improved = False
        # 2-opt pass
        i = 0
        while i < n - 2 and iteration < budget:
            a = tour[i]
            b = tour[i + 1]
            for c in nn_list[b]:
                if iteration >= budget:
                    break
                iteration += 1
                j = pos[c]
                if j <= i + 1 or j >= n - 1:
                    continue
                d = tour[(j + 1) % n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < -1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    for idx in range(i+1, j+1):
                        pos[tour[idx]] = idx
                    current_dist += delta
                    if current_dist < best_dist - 1e-12:
                        best_dist = current_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                        last_improvement = iteration
                    improved = True
                    break
            i += 1
        if iteration >= budget:
            break
        
        # If 2-opt didn't improve, try Or-opt
        if not improved:
            # Or-opt for segment lengths 2 and 3
            for L in [2, 3]:
                if iteration >= budget:
                    break
                for s in range(n):
                    if iteration >= budget:
                        break
                    # segment from s to s+L-1 (wrap around not handled, skip wrap)
                    if s + L > n:
                        continue
                    seg_indices = list(range(s, s+L))
                    seg_cities = tour[seg_indices].tolist()
                    # forward part before segment
                    part1 = tour[:s]
                    part2 = tour[s+L:]
                    # Try inserting at all positions
                    for k in range(len(part1) + len(part2) + 1):
                        if iteration >= budget:
                            break
                        iteration += 1
                        # new tour: part1[:k] + seg_cities + part1[k:] + part2?
                        # Actually consider adjacency: if k <= len(part1): insert into part1 at position k
                        # else: insert after part1, before part2
                        if k <= len(part1):
                            new_tour = np.concatenate((part1[:k], seg_cities, part1[k:], part2))
                        else:
                            new_tour = np.concatenate((part1, seg_cities, part2))
                        # compute delta efficiently? just compute full distance
                        new_dist = sum(distance_matrix[new_tour[i], new_tour[(i+1)%n]] for i in range(n))
                        if new_dist < current_dist - 1e-12:
                            tour = new_tour
                            pos = build_pos(tour)
                            current_dist = new_dist
                            if current_dist < best_dist - 1e-12:
                                best_dist = current_dist
                                best_tour = tour.copy()
                                report_best_tour(best_tour)
                                last_improvement = iteration
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # Restart if stagnant
        if iteration - last_improvement > restart_threshold and iteration < budget:
            new_start = np.random.randint(n)
            tour = regret_insertion(start=new_start)
            pos = build_pos(tour)
            current_dist = tour_distance(tour)
            if current_dist < best_dist - 1e-12:
                best_dist = current_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
            last_improvement = iteration
    
    return best_tour