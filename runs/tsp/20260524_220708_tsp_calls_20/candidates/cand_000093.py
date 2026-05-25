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
            chosen = candidates[rng.integers(len(candidates))]
            pos = best_data[chosen][2]
            tour.insert(pos, chosen)
            unvisited.remove(chosen)
        return np.array(tour, dtype=np.int64)

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
    threshold = max(5, n // 8)

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
            if no_improve_passes >= threshold:
                if ops >= budget:
                    break
                ops += 1
                L = max(2, n // 4)
                start_idx = rng.integers(0, n - L + 1)
                segment = set(cur_tour[start_idx:start_idx+L])
                base = [c for c in cur_tour if c not in segment]
                for city in segment:
                    best_cost = np.inf
                    best_pos = -1
                    m = len(base)
                    for i in range(m):
                        prev = base[i]
                        nxt = base[(i+1)%m]
                        inc = dist[prev, city] + dist[city, nxt] - dist[prev, nxt]
                        if inc < best_cost:
                            best_cost = inc
                            best_pos = i+1
                    base.insert(best_pos, city)
                new_tour = np.array(base, dtype=np.int64)
                new_dist = 0.0
                for k in range(n):
                    new_dist += dist[new_tour[k], new_tour[(k+1)%n]]
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_tour = new_tour.copy()
                    report_best_tour(best_tour)
                if new_dist <= cur_dist or new_dist <= best_dist * 1.05:
                    cur_tour = new_tour
                    cur_dist = new_dist
                else:
                    accept_prob = np.exp(-(new_dist - cur_dist) / max(1e-6, best_dist * 0.05))
                    if rng.random() < accept_prob:
                        cur_tour = new_tour
                        cur_dist = new_dist
                no_improve_passes = 0
            if no_improve_passes >= 2 * threshold:
                if ops >= budget:
                    break
                new_start = rng.integers(n)
                cur_tour = regret_construction(new_start)
                cur_dist = 0.0
                for i in range(n):
                    cur_dist += dist[cur_tour[i], cur_tour[(i+1)%n]]
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_tour = cur_tour.copy()
                    report_best_tour(best_tour)
                no_improve_passes = 0
    return best_tour