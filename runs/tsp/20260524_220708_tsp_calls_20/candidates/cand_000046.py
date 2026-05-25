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
    
    def compute_dist(tour):
        d = 0.0
        for i in range(n):
            d += distance_matrix[tour[i], tour[(i+1)%n]]
        return d
    
    def double_bridge_perturb(tour):
        n_ = len(tour)
        if n_ < 8:
            i, j = rng.choice(n_, 2, replace=False)
            tour[i], tour[j] = tour[j], tour[i]
            return tour
        # choose three breakpoints to create four segments
        a = rng.integers(1, n_-3)
        b = rng.integers(a+1, n_-2)
        c = rng.integers(b+1, n_-1)
        seg1 = tour[:a]
        seg2 = tour[a:b]
        seg3 = tour[b:c]
        seg4 = tour[c:]
        # reorder as 1,3,2,4
        new_tour = np.concatenate([seg1, seg3, seg2, seg4])
        return new_tour
    
    start = rng.integers(n)
    best_tour = regret_construction(start)
    best_dist = compute_dist(best_tour)
    report_best_tour(best_tour)
    cur_tour = best_tour.copy()
    cur_dist = best_dist
    ops = 0
    no_improve = 0
    perturb_count = 0
    max_no_improve = 5
    max_perturb = 10
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
                old = distance_matrix[a,b] + distance_matrix[c,d]
                new = distance_matrix[a,c] + distance_matrix[b,d]
                if new < old - 1e-12:
                    cur_tour[i+1:j+1] = cur_tour[i+1:j+1][::-1]
                    cur_dist = cur_dist - old + new
                    improved = True
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = cur_tour.copy()
                        report_best_tour(best_tour)
                    break
            if improved:
                break
        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve:
                # apply double-bridge perturbation
                cur_tour = double_bridge_perturb(cur_tour)
                cur_dist = compute_dist(cur_tour)
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_tour = cur_tour.copy()
                    report_best_tour(best_tour)
                no_improve = 0
                perturb_count += 1
                if perturb_count >= max_perturb:
                    # full restart
                    start = rng.integers(n)
                    cur_tour = regret_construction(start)
                    cur_dist = compute_dist(cur_tour)
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = cur_tour.copy()
                        report_best_tour(best_tour)
                    perturb_count = 0
        else:
            no_improve = 0
    return best_tour