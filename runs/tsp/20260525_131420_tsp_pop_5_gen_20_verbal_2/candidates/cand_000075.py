import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    dist = distance_matrix
    n = dist.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    
    K = min(30, n - 1)
    neighbor_lists = np.argsort(dist, axis=1)[:, 1:K+1]
    
    def regret_insertion(perturb=False):
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            best_costs = {}
            for city in unvisited:
                min_cost = float('inf')
                min_pos = -1
                second_min = float('inf')
                for i in range(len(tour)):
                    a = tour[i]
                    b = tour[(i+1) % len(tour)]
                    cost = dist[a, city] + dist[city, b] - dist[a, b]
                    if cost < min_cost:
                        second_min = min_cost
                        min_cost = cost
                        min_pos = i + 1
                    elif cost < second_min:
                        second_min = cost
                best_costs[city] = (min_cost, min_pos, second_min)
            if len(unvisited) == 1:
                city = next(iter(unvisited))
            else:
                if perturb and np.random.random() < 0.5:
                    city = np.random.choice(list(unvisited))
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
        return np.array(tour, dtype=np.int32)
    
    def compute_total(t):
        s = 0.0
        for i in range(n):
            s += dist[t[i], t[(i+1)%n]]
        return s
    
    tour = regret_insertion(perturb=False)
    best_tour = tour.copy()
    best_dist = compute_total(tour)
    report_best_tour(best_tour)
    
    pos = np.empty(n, dtype=int)
    def update_pos(t):
        for idx, city in enumerate(t):
            pos[city] = idx
    update_pos(tour)
    total = best_dist
    
    iteration = 0
    restart_threshold = max(1, int(budget * 0.15))
    no_improve = 0
    
    while iteration < budget:
        # Best-improvement 2-opt with candidate lists
        best_delta = 0.0
        best_i = best_j = -1
        for i in range(n - 2):
            if iteration >= budget:
                break
            a = tour[i]
            b = tour[i+1]
            for nb in neighbor_lists[b]:
                j = pos[nb]
                if j <= i + 1:
                    continue
                c = tour[j]
                d = tour[(j+1) % n]
                delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                iteration += 1
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        if best_delta < -1e-12:
            i, j = best_i, best_j
            tour[i+1:j+1] = tour[i+1:j+1][::-1]
            update_pos(tour)
            total += best_delta
            if total < best_dist - 1e-12:
                best_dist = total
                best_tour = tour.copy()
                report_best_tour(best_tour)
            no_improve = 0
        else:
            no_improve += 1
        if no_improve > restart_threshold and iteration < budget:
            # Restart with perturbed regret insertion + random 2-opt swap
            tour = regret_insertion(perturb=True)
            # Apply a random 2-opt swap for further diversity
            i = np.random.randint(n - 2)
            j = np.random.randint(i+2, n)
            if np.random.random() < 0.5:
                tour[i+1:j+1] = tour[i+1:j+1][::-1]
            update_pos(tour)
            total = compute_total(tour)
            no_improve = 0
    
    return best_tour