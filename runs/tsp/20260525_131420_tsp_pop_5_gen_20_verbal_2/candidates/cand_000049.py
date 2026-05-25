import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour)
        return tour

    def regret_insertion():
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            best = {}
            for city in unvisited:
                min_cost = float('inf')
                min_pos = -1
                second_min = float('inf')
                for i in range(len(tour)):
                    a = tour[i]
                    b = tour[(i+1)%len(tour)]
                    cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                    if cost < min_cost:
                        second_min = min_cost
                        min_cost = cost
                        min_pos = i+1
                    elif cost < second_min:
                        second_min = cost
                best[city] = (min_cost, min_pos, second_min)
            if len(unvisited) == 1:
                city = next(iter(unvisited))
            else:
                max_regret = -1
                city = None
                for c, (cost, pos, second) in best.items():
                    regret = second - cost
                    if regret > max_regret:
                        max_regret = regret
                        city = c
            cost, pos, _ = best[city]
            tour.insert(pos, city)
            unvisited.remove(city)
        return np.array(tour, dtype=np.int32)

    def double_bridge(t):
        n_ = len(t)
        a = np.random.randint(0, n_//2)
        b = np.random.randint(a+1, n_//2 + n_//4)
        c = np.random.randint(b+1, min(n_-1, b + n_//4))
        d = np.random.randint(c+1, n_)
        if d >= n_:
            d = n_-1
        if a >= b or b >= c or c >= d:
            return t.copy()
        new_tour = np.concatenate([t[:a], t[c:d], t[b:c], t[a:b], t[d:]])
        return new_tour.astype(np.int32)

    tour = regret_insertion()
    total = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    best_tour = tour.copy()
    best_dist = total
    report_best_tour(best_tour)

    attempts = 0
    no_improve = 0
    threshold = max(50, budget // 20)
    while attempts < budget:
        improved = False
        for i in range(n-2):
            if attempts >= budget:
                break
            for j in range(i+2, n):
                if attempts >= budget:
                    break
                attempts += 1
                a = tour[i]
                b = tour[i+1]
                c = tour[j]
                d = tour[(j+1)%n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < -1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    total += delta
                    if total < best_dist - 1e-12:
                        best_dist = total
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                        no_improve = 0
                    improved = True
                    break
            if improved:
                break
        if not improved:
            no_improve += 1
            if no_improve >= threshold and attempts < budget:
                tour = double_bridge(tour)
                total = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
                attempts += 1
                no_improve = 0
        else:
            no_improve = 0
    return best_tour