import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        np.random.default_rng(seed).shuffle(tour)
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
    
    cur_dist = best_dist
    ops = 0
    no_improve_threshold = 3
    no_improve_count = 0
    
    while ops < budget:
        improved = False
        for i in range(n-1):
            if ops >= budget:
                break
            for j in range(i+2, n):
                if ops >= budget:
                    break
                ops += 1
                a, b = tour[i], tour[(i+1)%n]
                c, d = tour[j], tour[(j+1)%n]
                old = distance_matrix[a,b] + distance_matrix[c,d]
                new = distance_matrix[a,c] + distance_matrix[b,d]
                delta = new - old
                if delta < -1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    cur_dist += delta
                    improved = True
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break
            if improved:
                break
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= no_improve_threshold:
                no_improve_count = 0
                start = rng.integers(n)
                tour = regret_construction(start)
                cur_dist = 0.0
                for i in range(n):
                    cur_dist += distance_matrix[tour[i], tour[(i+1)%n]]
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
    return best_tour