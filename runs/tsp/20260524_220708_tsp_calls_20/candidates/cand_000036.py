import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        rng = np.random.default_rng(seed)
        rng.shuffle(tour)
        report_best_tour(tour)
        return tour
    rng = np.random.default_rng(seed)

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
                    inc = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
                    if inc < best_cost:
                        second_best = best_cost
                        best_cost = inc
                        best_pos = i+1
                    elif inc < second_best:
                        second_best = inc
                regret = second_best - best_cost if second_best != np.inf else best_cost
                best_data[city] = (regret, best_pos, best_cost)
            chosen = max(unvisited, key=lambda c: best_data[c][0])
            pos = best_data[chosen][1]
            tour.insert(pos, chosen)
            unvisited.remove(chosen)
        return np.array(tour, dtype=np.int64)

    start = rng.integers(n)
    tour = regret_construction(start)
    best_tour = tour.copy()
    best_dist = 0.0
    for i in range(n):
        best_dist += distance_matrix[tour[i], tour[(i+1)%n]]
    report_best_tour(best_tour)

    ops = 0
    improved = True
    while ops < budget:
        if not improved:
            # Perturbation: random segment relocation (Or-opt like)
            max_len = max(1, n//3)
            seg_len = rng.integers(1, max_len+1)
            seg_start = rng.integers(0, n)
            seg_indices = [(seg_start + i) % n for i in range(seg_len)]
            seg = tour[seg_indices]
            remaining = np.delete(tour, seg_indices)
            insert_pos = rng.integers(0, len(remaining)+1)
            tour = np.concatenate([remaining[:insert_pos], seg, remaining[insert_pos:]])
            ops += 1
            cur_dist = 0.0
            for k in range(n):
                cur_dist += distance_matrix[tour[k], tour[(k+1)%n]]
            if cur_dist < best_dist:
                best_dist = cur_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
            improved = True
            continue
        improved = False
        # 2-opt first improvement
        for i in range(n-1):
            if ops >= budget:
                break
            for j in range(i+2, n):
                if ops >= budget:
                    break
                ops += 1
                a = tour[i]; b = tour[(i+1)%n]
                c = tour[j]; d = tour[(j+1)%n]
                old = distance_matrix[a,b] + distance_matrix[c,d]
                new = distance_matrix[a,c] + distance_matrix[b,d]
                if new < old - 1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    cur_dist = 0.0
                    for k in range(n):
                        cur_dist += distance_matrix[tour[k], tour[(k+1)%n]]
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break
            if improved:
                break

    # Post-optimization: limited random 3-opt moves
    if ops < budget:
        for _ in range(min(5, budget - ops)):
            i = rng.integers(0, n-2)
            j = rng.integers(i+2, n-1)
            k = rng.integers(j+2, n)
            A = tour[:i+1]
            B = tour[i+1:j+1]
            C = tour[j+1:k+1]
            D = tour[k+1:]
            perm = rng.integers(1, 4)
            if perm == 1:
                new_tour = np.concatenate([A, C, B, D])
            elif perm == 2:
                new_tour = np.concatenate([A, B, D, C])
            else:
                new_tour = np.concatenate([A, D, C, B])
            ops += 1
            cur_dist = 0.0
            for l in range(n):
                cur_dist += distance_matrix[new_tour[l], new_tour[(l+1)%n]]
            if cur_dist < best_dist:
                best_dist = cur_dist
                best_tour = new_tour.copy()
                report_best_tour(best_tour)
                tour = new_tour
    return best_tour