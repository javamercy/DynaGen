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
            max_regret = max(d[0] for d in best_data.values())
            candidates = [c for c, v in best_data.items() if v[0] == max_regret]
            chosen = min(candidates, key=lambda c: best_data[c][1])
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
    pass_counter = 0
    threshold = 2  # consecutive non-improving passes before restart

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
            if improved or ops >= budget:
                break
        if ops >= budget:
            break

        if improved:
            pass_counter = 0
        else:
            pass_counter += 1

        if pass_counter >= threshold:
            # partial restart: remove random segment and reinsert greedily
            L = rng.integers(1, max(2, n//3)+1)
            start_idx = rng.integers(0, n-L+1)
            segment = set(cur_tour[start_idx:start_idx+L])
            base = [c for c in cur_tour if c not in segment]
            # reinsert in order they appear in segment
            seg_order = [c for c in cur_tour if c in segment]
            for city in seg_order:
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
            cur_tour = np.array(base, dtype=np.int64)
            cur_dist = 0.0
            for k in range(n):
                cur_dist += dist[cur_tour[k], cur_tour[(k+1)%n]]
            if cur_dist < best_dist:
                best_dist = cur_dist
                best_tour = cur_tour.copy()
                report_best_tour(best_tour)
            pass_counter = 0

    return best_tour