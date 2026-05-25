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
    cur_dist = best_dist
    ops = 0
    ops_since_restart = 0
    stall_ops = max(10, 5 * n)

    while ops < budget:
        improved = False
        for i in range(n-1):
            if ops >= budget:
                break
            for j in range(i+2, n):
                if ops >= budget:
                    break
                ops += 1
                ops_since_restart += 1
                a, b = tour[i], tour[(i+1)%n]
                c, d = tour[j], tour[(j+1)%n]
                old = dist[a,b] + dist[c,d]
                new = dist[a,c] + dist[b,d]
                if new < old - 1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    cur_dist += new - old
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    improved = True
                    ops_since_restart = 0
                    break
            if improved:
                break

        if not improved:
            if ops_since_restart >= stall_ops and ops < budget:
                L = rng.integers(max(1, n//4), max(2, n//2)+1)
                start_idx = rng.integers(0, n-L+1)
                segment = set(tour[start_idx:start_idx+L])
                base = [c for c in tour if c not in segment]
                tour_list = base[:]
                # Regret reinsertion for segment cities
                unvisited = list(segment)
                while unvisited:
                    best_data = {}
                    for city in unvisited:
                        best_cost = np.inf
                        second_best = np.inf
                        best_pos = -1
                        m = len(tour_list)
                        for i in range(m):
                            prev = tour_list[i]
                            nxt = tour_list[(i+1)%m]
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
                    tour_list.insert(pos, chosen)
                    unvisited.remove(chosen)
                tour = np.array(tour_list, dtype=np.int64)
                cur_dist = 0.0
                for i in range(n):
                    cur_dist += dist[tour[i], tour[(i+1)%n]]
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                ops_since_restart = 0
            else:
                pass

        if ops >= budget:
            break

    return best_tour