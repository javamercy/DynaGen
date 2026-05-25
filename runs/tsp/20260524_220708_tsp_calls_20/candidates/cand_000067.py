import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        np.random.default_rng(seed).shuffle(tour)
        return tour

    rng = np.random.default_rng(seed)
    dist = distance_matrix

    def regret_construction(start):
        tour = [start]
        unvisited = set(range(n)) - {start}
        while unvisited:
            best_data = {}
            for city in unvisited:
                best_cost = np.inf
                second_best = np.inf
                best_pos = -1
                m = len(tour)
                for i in range(m):
                    prev = tour[i]
                    nxt = tour[(i+1) % m]
                    inc = dist[prev, city] + dist[city, nxt] - dist[prev, nxt]
                    if inc < best_cost:
                        second_best = best_cost
                        best_cost = inc
                        best_pos = i+1
                    elif inc < second_best:
                        second_best = inc
                regret = second_best - best_cost if second_best != np.inf else best_cost
                best_data[city] = (regret, best_cost, best_pos)
            max_regret = max(v[0] for v in best_data.values())
            candidates = [c for c, v in best_data.items() if v[0] == max_regret]
            chosen = min(candidates, key=lambda c: best_data[c][1])
            pos = best_data[chosen][2]
            tour.insert(pos, chosen)
            unvisited.remove(chosen)
        return np.array(tour, dtype=np.int64)

    def regret_insert(base, to_insert):
        # base is a list representing a partial tour (cyclic)
        # to_insert is a set of city indices
        base_list = list(base)
        while to_insert:
            best_data = {}
            for city in to_insert:
                best_cost = np.inf
                second_best = np.inf
                best_pos = -1
                m = len(base_list)
                for i in range(m + 1):
                    prev = base_list[i-1] if i > 0 else base_list[-1]
                    nxt = base_list[i] if i < m else base_list[0]
                    inc = dist[prev, city] + dist[city, nxt] - dist[prev, nxt]
                    if inc < best_cost:
                        second_best = best_cost
                        best_cost = inc
                        best_pos = i
                    elif inc < second_best:
                        second_best = inc
                regret = second_best - best_cost if second_best != np.inf else best_cost
                best_data[city] = (regret, best_cost, best_pos)
            max_regret = max(v[0] for v in best_data.values())
            candidates = [c for c, v in best_data.items() if v[0] == max_regret]
            chosen = min(candidates, key=lambda c: best_data[c][1])
            pos = best_data[chosen][2]
            base_list.insert(pos, chosen)
            to_insert.remove(chosen)
        return np.array(base_list, dtype=np.int64)

    start = rng.integers(n)
    tour = regret_construction(start)
    best_tour = tour.copy()
    best_dist = 0.0
    for i in range(n):
        best_dist += dist[tour[i], tour[(i+1)%n]]
    report_best_tour(best_tour)
    cur_tour = tour.copy()
    cur_dist = best_dist
    ops = 0
    no_improve_passes = 0
    threshold = max(3, n // 10)

    while ops < budget:
        improved = False
        for i in range(n-1):
            if ops >= budget:
                break
            for j in range(i+2, n):
                if ops >= budget:
                    break
                ops += 1
                a, b = cur_tour[i], cur_tour[(i+1)%n]
                c, d = cur_tour[j], cur_tour[(j+1)%n]
                old = dist[a,b] + dist[c,d]
                new = dist[a,c] + dist[b,d]
                if new < old - 1e-12:
                    cur_tour[i+1:j+1] = cur_tour[i+1:j+1][::-1]
                    cur_dist += new - old
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = cur_tour.copy()
                        report_best_tour(best_tour)
                    improved = True
                    break
            if improved:
                break
        if ops >= budget:
            break
        if improved:
            no_improve_passes = 0
        else:
            no_improve_passes += 1
            if no_improve_passes >= threshold and ops < budget:
                ops += 1
                # Large segment removal and regret reinsertion
                L = rng.integers(max(2, n//4), max(3, n//2)+1)
                if L > n:
                    L = n
                start_idx = rng.integers(0, n - L + 1)
                segment = set(cur_tour[start_idx:start_idx+L])
                base = [c for c in cur_tour if c not in segment]
                new_tour = regret_insert(base, segment)
                new_dist = 0.0
                for k in range(n):
                    new_dist += dist[new_tour[k], new_tour[(k+1)%n]]
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_tour = new_tour.copy()
                    report_best_tour(best_tour)
                cur_tour = new_tour
                cur_dist = new_dist
                no_improve_passes = 0
    return best_tour