import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour)
        return tour
    rng = np.random.RandomState(seed)

    def tour_length(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))

    def regret_insertion():
        start = rng.randint(n)
        dists = distance_matrix[start].copy()
        dists[start] = np.inf
        second = np.argmin(dists)
        tour = [start, second]
        remaining = set(range(n)) - {start, second}
        while remaining:
            best_regret = -1e100
            best_city = None
            best_pos = None
            best_cost = None
            for city in remaining:
                costs = []
                L = len(tour)
                for p in range(L):
                    left = tour[p]
                    right = tour[(p+1)%L]
                    cost = distance_matrix[left, city] + distance_matrix[city, right] - distance_matrix[left, right]
                    costs.append(cost)
                sorted_costs = sorted(costs)
                best = sorted_costs[0]
                second_best = sorted_costs[1] if len(sorted_costs) > 1 else best
                regret = second_best - best
                if regret > best_regret or (regret == best_regret and (best_cost is None or best < best_cost)):
                    best_regret = regret
                    best_city = city
                    best_pos = np.argmin(costs)
                    best_cost = best
            insert_idx = best_pos + 1
            tour = tour[:insert_idx] + [best_city] + tour[insert_idx:]
            remaining.remove(best_city)
        return np.array(tour, dtype=np.int32)

    best_tour = None
    best_len = np.inf
    visited = set()
    max_restarts = 3
    if budget < 100:
        max_restarts = 1

    for _ in range(max_restarts):
        tour = regret_insertion()
        tour_len = tour_length(tour)
        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour.copy()
            report_best_tour(best_tour.copy())

        attempts = 0
        improved = True
        no_improve_passes = 0
        while attempts < budget:
            if not improved:
                no_improve_passes += 1
                if no_improve_passes >= 1 and budget - attempts >= n:
                    # apply random 4-opt (double bridge) kick
                    idx = sorted(rng.choice(n, 4, replace=False))
                    a, b, c, d = idx[0], idx[1], idx[2], idx[3]
                    tour = np.concatenate([tour[:a+1], tour[c+1:d+1], tour[b+1:c+1], tour[a+1:b+1], tour[d+1:]])
                    tour_len = tour_length(tour)
                    improved = True
                    no_improve_passes = 0
                    continue
                else:
                    break
            improved = False
            order = list(range(n))
            rng.shuffle(order)
            for i_idx in range(n):
                i = order[i_idx]
                for j_idx in range(i_idx+2, n):
                    j = order[j_idx]
                    if i > j:
                        i, j = j, i
                    if j - i <= 1 or (i == 0 and j == n-1):
                        continue
                    a = tour[i]
                    b = tour[(i+1)%n]
                    c = tour[j]
                    d = tour[(j+1)%n]
                    old = distance_matrix[a,b] + distance_matrix[c,d]
                    new = distance_matrix[a,c] + distance_matrix[b,d]
                    attempts += 1
                    if attempts > budget:
                        break
                    if new < old:
                        # apply 2-opt
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        tour_len += new - old
                        if tour_len < best_len:
                            best_len = tour_len
                            best_tour = tour.copy()
                            report_best_tour(best_tour.copy())
                        improved = True
                        no_improve_passes = 0
                        break
                    else:
                        # with small probability accept worsening move
                        if rng.rand() < 0.02:
                            tour[i+1:j+1] = tour[i+1:j+1][::-1]
                            tour_len += new - old
                            improved = True
                            no_improve_passes = 0
                            break
                if improved or attempts >= budget:
                    break
            if attempts >= budget:
                break

    return best_tour