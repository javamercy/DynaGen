import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    
    # Precompute nearest neighbor lists (k = min(n-1, 50))
    k = min(n-1, 50)
    nn_list = []
    for i in range(n):
        order = np.argsort(distance_matrix[i])
        nn_list.append(order[1:k+1] if n > 1 else [])
    
    # Build position array from tour
    def build_pos(tour):
        pos = np.empty(n, dtype=int)
        for idx, city in enumerate(tour):
            pos[city] = idx
        return pos
    
    # Compute full tour distance
    def compute_dist(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))
    
    # Regret insertion with random start and random tie-breaking
    def regret_insertion(start=None):
        if start is None:
            start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            best = []
            for city in unvisited:
                min_cost = float('inf')
                min_pos = -1
                second_min = float('inf')
                for i in range(len(tour)):
                    a = tour[i]
                    b = tour[(i+1) % len(tour)]
                    cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                    if cost < min_cost:
                        second_min = min_cost
                        min_cost = cost
                        min_pos = i+1
                    elif cost < second_min:
                        second_min = cost
                regret = second_min - min_cost
                best.append((city, min_cost, min_pos, regret))
            max_regret = max(c[3] for c in best)
            candidates = [c for c in best if c[3] == max_regret]
            city, _, pos, _ = candidates[np.random.randint(len(candidates))]
            tour.insert(pos, city)
            unvisited.remove(city)
        return np.array(tour, dtype=np.int32)
    
    # Build initial tour with multiple restarts (up to 5)
    num_restarts = min(5, max(1, budget // 5))
    best_tour = None
    best_dist = float('inf')
    for _ in range(num_restarts):
        tour = regret_insertion()
        dist = compute_dist(tour)
        if dist < best_dist - 1e-12:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
    tour = best_tour.copy()
    pos = build_pos(tour)
    local_dist = best_dist
    
    iteration = 0
    last_improvement = 0
    restart_threshold = max(10, budget // 10)
    
    while iteration < budget:
        # 2-opt improvement phase
        improved = True
        while improved and iteration < budget:
            improved = False
            # Iterate over all edges in random order? Use sequential order.
            for i in range(n-2):
                if iteration >= budget:
                    break
                a = tour[i]
                b = tour[i+1]
                # Consider candidate successors of b
                for c in nn_list[b]:
                    if iteration >= budget:
                        break
                    j = pos[c]
                    if j <= i+1 or j >= n-1:
                        continue
                    d = tour[(j+1) % n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    iteration += 1
                    if delta < -1e-12:
                        # Apply 2-opt move
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        # Update positions
                        for idx in range(i+1, j+1):
                            pos[tour[idx]] = idx
                        local_dist += delta
                        if local_dist < best_dist - 1e-12:
                            best_dist = local_dist
                            best_tour = tour.copy()
                            report_best_tour(best_tour)
                            last_improvement = iteration
                        improved = True
                        break
                if improved:
                    break
            # end for i
        # end while improved
        
        # Stagnation check and restart
        if iteration >= budget:
            break
        if iteration - last_improvement > restart_threshold:
            # restart from new random start
            tour = regret_insertion(start=np.random.randint(n))
            pos = build_pos(tour)
            local_dist = compute_dist(tour)
            if local_dist < best_dist - 1e-12:
                best_dist = local_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
            last_improvement = iteration
        else:
            # no improvement but not yet stagnated; just continue loop
            pass
        # To avoid infinite loop, increment iteration? Actually iteration is incremented in inner loop.
        # If we didn't have any moves, we need to break? But we are in outer while loop, so it will continue.
        # But if no moves attempted, iteration hasn't increased, so we force an increment.
        if not improved and iteration < budget:
            # no move attempted (e.g., all edges checked but no improvement) - just count a dummy iteration
            iteration += 1
    
    return best_tour