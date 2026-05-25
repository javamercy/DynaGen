import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    dist = distance_matrix
    n = dist.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    K = min(50, n - 1)
    neighbor_lists = np.argsort(dist, axis=1)[:, 1:K+1]

    def regret_insertion(perturb=False):
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            if len(unvisited) == 1:
                city = next(iter(unvisited))
                # compute best insertion position
                best_cost = float('inf')
                best_pos = 0
                for i in range(len(tour)):
                    a = tour[i]
                    b = tour[(i+1) % len(tour)]
                    cost = dist[a, city] + dist[city, b] - dist[a, b]
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = i + 1
                tour.insert(best_pos, city)
                unvisited.remove(city)
                break
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
            if perturb and np.random.random() < 0.3:
                city = np.random.choice(list(unvisited))
            else:
                max_regret = -1
                candidates = []
                for c, (cost, pos, second) in best_costs.items():
                    regret = second - cost
                    if regret > max_regret:
                        max_regret = regret
                        candidates = [c]
                    elif regret == max_regret:
                        candidates.append(c)
                city = np.random.choice(candidates)
            cost, pos, _ = best_costs[city]
            tour.insert(pos, city)
            unvisited.remove(city)
        return np.array(tour, dtype=np.int32)

    tour = regret_insertion(perturb=False)
    pos = np.empty(n, dtype=int)
    for idx, city in enumerate(tour):
        pos[city] = idx
    total = sum(dist[tour[i], tour[(i+1) % n]] for i in range(n))
    best_tour = tour.copy()
    best_dist = total
    report_best_tour(best_tour)

    iteration = 0
    restart_threshold = max(1, budget // 10)
    no_improve = 0

    while iteration < budget:
        improved = False
        i = 0
        while i < n - 2 and iteration < budget:
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
                if iteration >= budget:
                    break
                if delta < -1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    for idx in range(i+1, j+1):
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
            i += 1
        if improved:
            no_improve = 0
        else:
            no_improve += 1
        if no_improve > restart_threshold and iteration < budget:
            tour = regret_insertion(perturb=True)
            for idx, city in enumerate(tour):
                pos[city] = idx
            total = sum(dist[tour[i], tour[(i+1) % n]] for i in range(n))
            if total < best_dist - 1e-12:
                best_dist = total
                best_tour = tour.copy()
                report_best_tour(best_tour)
            no_improve = 0

    return best_tour