import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    dist = distance_matrix
    n = dist.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    
    # Precompute nearest neighbors (top K) for each city
    K = min(30, n - 1)
    neighbor_lists = np.argsort(dist, axis=1)[:, 1:K+1]  # exclude self
    
    def regret_insertion(perturb=False):
        start = np.random.randint(n)
        tour_list = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            best_costs = {}
            for city in unvisited:
                min_cost = float('inf')
                min_pos = -1
                second_min = float('inf')
                for i in range(len(tour_list)):
                    a = tour_list[i]
                    b = tour_list[(i + 1) % len(tour_list)]
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
                if perturb and np.random.random() < 0.3:
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
            tour_list.insert(pos, city)
            unvisited.remove(city)
        return np.array(tour_list, dtype=np.int32)
    
    # Initial tour
    tour = regret_insertion(perturb=False)
    pos = np.empty(n, dtype=int)
    for idx, city in enumerate(tour):
        pos[city] = idx
    total = sum(dist[tour[i], tour[(i + 1) % n]] for i in range(n))
    best_tour = tour.copy()
    best_dist = total
    report_best_tour(best_tour)
    
    iteration = 0
    restart_threshold = max(1, int(budget * 0.15))
    no_improve = 0
    
    while iteration < budget:
        improved = False
        for i in range(n - 2):
            if iteration >= budget:
                break
            a = tour[i]
            b = tour[i + 1]
            for nb in neighbor_lists[b]:
                j = pos[nb]
                if j <= i + 1:
                    continue
                c = tour[j]
                d = tour[(j + 1) % n]
                delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                iteration += 1
                if delta < -1e-12:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    for idx in range(i + 1, j + 1):
                        pos[tour[idx]] = idx
                    total += delta
                    if total < best_dist - 1e-12:
                        best_dist = total
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    improved = True
                    break
            if improved:
                break
        if improved:
            no_improve = 0
        else:
            no_improve += 1
        if no_improve > restart_threshold and iteration < budget:
            # Restart with perturbed regret insertion
            tour = regret_insertion(perturb=True)
            for idx, city in enumerate(tour):
                pos[city] = idx
            total = sum(dist[tour[i], tour[(i + 1) % n]] for i in range(n))
            no_improve = 0
    
    return best_tour