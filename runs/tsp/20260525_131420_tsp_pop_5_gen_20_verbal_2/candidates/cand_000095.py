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
    
    def regret_insertion():
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
    restart_threshold = max(10, budget // 10)
    
    while iteration < budget:
        improved = False
        # 2-opt
        for i in range(n - 2):
            if iteration >= budget:
                break
            iteration += 1
            a = tour[i]
            b = tour[i + 1]
            for c in nn_list[b]:
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
            if improved:
                break
        if iteration >= budget:
            break
        
        # limited 3-opt: moving a segment of length 2
        if not improved:
            for i in range(n - 3):
                if iteration >= budget:
                    break
                iteration += 1
                # segment: positions i and i+1 (two cities)
                seg = [tour[i], tour[i+1]]
                # current edges: (tour[i-1], seg[0]), (seg[1], tour[i+2]), and at insertion point
                # compute delta for each possible insertion point j (not overlapping)
                for j in range(n - 2):
                    if j == i or j == i-1 or j == i+1:  # skip overlapping
                        continue
                    # compute delta
                    # pre-removal: edges (tour[i-1], seg[0]), (seg[1], tour[i+2]), (tour[j], tour[(j+1)%n])
                    a = tour[i-1] if i > 0 else tour[-1]
                    b = seg[0]
                    c = seg[1]
                    d = tour[(i+2)%n]
                    e = tour[j]
                    f = tour[(j+1)%n]
                    # after removal and insertion: edges (a,d), (e, b), (c, f)
                    current_edges = distance_matrix[a, b] + distance_matrix[c, d] + distance_matrix[e, f]
                    new_edges = distance_matrix[a, d] + distance_matrix[e, b] + distance_matrix[c, f]
                    delta = new_edges - current_edges
                    if delta < -1e-12:
                        # apply move: remove segment and insert at j
                        # build new tour
                        tour = np.delete(tour, [i, i+1])
                        # adjust j for shift
                        if j > i:
                            j -= 2
                        tour = np.insert(tour, j+1, seg)
                        pos = build_pos(tour)
                        current_dist += delta
                        if current_dist < best_dist - 1e-12:
                            best_dist = current_dist
                            best_tour = tour.copy()
                            report_best_tour(best_tour)
                            last_improvement = iteration
                        improved = True
                        break
                if improved:
                    break
        
        # restart if stagnant
        if iteration - last_improvement > restart_threshold and iteration < budget:
            new_start = np.random.randint(n)
            tour = regret_insertion()
            pos = build_pos(tour)
            current_dist = tour_distance(tour)
            if current_dist < best_dist - 1e-12:
                best_dist = current_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
            last_improvement = iteration
    
    return best_tour