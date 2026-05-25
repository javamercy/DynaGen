import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        np.random.default_rng(seed).shuffle(tour)
        return tour
    rng = np.random.default_rng(seed)
    dist = distance_matrix

    def nn_construction(start):
        tour = [start]
        unvisited = set(range(n)) - {start}
        last = start
        while unvisited:
            nearest = min(unvisited, key=lambda c: dist[last, c])
            tour.append(nearest)
            unvisited.remove(nearest)
            last = nearest
        return np.array(tour, dtype=np.int64)

    start = rng.integers(n)
    tour = nn_construction(start)
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
            if ops_since_restart >= stall_ops:
                L = rng.integers(1, max(2, n//3)+1)
                start_idx = rng.integers(0, n-L+1)
                segment = set(tour[start_idx:start_idx+L])
                base = [c for c in tour if c not in segment]
                tour_list = base[:]
                for city in segment:
                    best_cost = np.inf
                    best_pos = -1
                    m = len(tour_list)
                    for i in range(m):
                        prev = tour_list[i]
                        nxt = tour_list[(i+1)%m]
                        inc = dist[prev, city] + dist[city, nxt] - dist[prev, nxt]
                        if inc < best_cost:
                            best_cost = inc
                            best_pos = i+1
                    tour_list.insert(best_pos, city)
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
        else:
            ops_since_restart = 0

        if ops >= budget:
            break

    return best_tour