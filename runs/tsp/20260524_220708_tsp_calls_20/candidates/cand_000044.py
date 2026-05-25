import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        np.random.default_rng(seed).shuffle(tour)
        return tour
    rng = np.random.default_rng(seed)
    # Cheapest insertion construction
    start = rng.integers(n)
    tour = [start]
    # find nearest neighbor to start
    nearest = min((i for i in range(n) if i != start), key=lambda i: distance_matrix[start, i])
    tour.append(nearest)
    unvisited = set(range(n)) - {start, nearest}
    while unvisited:
        best_city = None
        best_pos = -1
        best_inc = np.inf
        for city in unvisited:
            for pos in range(len(tour) + 1):
                prev = tour[(pos - 1) % len(tour)]
                nxt = tour[pos % len(tour)] if pos < len(tour) else tour[0]
                inc = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
                if inc < best_inc:
                    best_inc = inc
                    best_city = city
                    best_pos = pos
        tour.insert(best_pos, best_city)
        unvisited.remove(best_city)
    tour = np.array(tour, dtype=np.int64)
    best_dist = 0.0
    for i in range(n):
        best_dist += distance_matrix[tour[i], tour[(i+1)%n]]
    best_tour = tour.copy()
    report_best_tour(best_tour)
    ops = 0
    improved = True
    stagnation = 0
    threshold = 1
    while ops < budget:
        if not improved:
            stagnation += 1
            threshold = 1 + int((ops / budget) * 9) if budget > 0 else 10
            if stagnation > threshold:
                if ops >= budget:
                    break
                ops += 1
                cuts = sorted(rng.choice(range(1, n), size=4, replace=False))
                a, b, c, d = cuts
                tour = np.concatenate([tour[:a], tour[c:d], tour[b:c], tour[a:b], tour[d:]])
                improved = True
                stagnation = 0
                cur_dist = 0.0
                for i in range(n):
                    cur_dist += distance_matrix[tour[i], tour[(i+1)%n]]
                if cur_dist < best_dist - 1e-12:
                    best_dist = cur_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                continue
        else:
            stagnation = 0
        improved = False
        # 2-opt first-improvement
        for i in range(n-1):
            if ops >= budget:
                break
            for j in range(i+2, n):
                if ops >= budget:
                    break
                ops += 1
                a = tour[i]
                b = tour[(i+1)%n]
                c = tour[j]
                d = tour[(j+1)%n]
                old = distance_matrix[a,b] + distance_matrix[c,d]
                new = distance_matrix[a,c] + distance_matrix[b,d]
                if new < old - 1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    cur_dist = best_dist + (new - old)
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break
            if improved:
                break
        if improved:
            continue
        # node insertion (3-opt) first-improvement
        for i in range(n):
            if ops >= budget:
                break
            for j in range(n-1):
                if ops >= budget:
                    break
                if j == i or j == (i-1+n)%n:
                    continue
                ops += 1
                prev = tour[(i-1+n)%n]
                nxt = tour[(i+1)%n]
                j_nxt = tour[(j+1)%n]
                old_edges = distance_matrix[prev, tour[i]] + distance_matrix[tour[i], nxt] + distance_matrix[tour[j], j_nxt]
                new_edges = distance_matrix[prev, nxt] + distance_matrix[tour[j], tour[i]] + distance_matrix[tour[i], j_nxt]
                if new_edges < old_edges - 1e-12:
                    lst = tour.tolist()
                    node = lst.pop(i)
                    lst.insert(j+1, node)
                    tour = np.array(lst, dtype=np.int64)
                    improved = True
                    cur_dist = best_dist + (new_edges - old_edges)
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break
            if improved:
                break
    return best_tour