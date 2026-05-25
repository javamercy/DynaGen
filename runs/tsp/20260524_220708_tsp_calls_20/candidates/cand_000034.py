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
    
    def compute_dist(t):
        d = 0.0
        for i in range(n):
            d += distance_matrix[t[i], t[(i+1)%n]]
        return d
    
    def double_bridge(t):
        # Random 4-cut: cut at pos p1, p2, p3, p4 and reconnect in different order
        p = sorted(rng.choice(range(n), 4, replace=False))
        p1, p2, p3, p4 = p
        # segments: [p2:p3], [p4:] + [:p1], [p1:p2], [p3:p4]? We'll use standard double-bridge
        # Actually common: cut at p1,p2,p3,p4 then reconnect as (p1->p3, p3->p2, p2->p4, p4->p1)? 
        # Let's do: segments A=[p1:p2], B=[p2:p3], C=[p3:p4], D=[p4:] + [:p1]
        # New tour = A reversed? Double bridge usually: A, C, B, D but with reversal of B and C?
        # Simplest: create new order: tour[0:p1], tour[p3:p4], tour[p2:p3], tour[p1:p2], tour[p4:]
        # But careful with wrapping. We'll concatenate lists for simplicity.
        t_list = t.tolist()
        a = t_list[p1:p2]
        b = t_list[p2:p3]
        c = t_list[p3:p4]
        d = t_list[p4:] + t_list[:p1]
        new_t = a + c + b + d
        return np.array(new_t, dtype=np.int64)
    
    start = rng.integers(n)
    tour = regret_construction(start)
    best_tour = tour.copy()
    best_dist = compute_dist(tour)
    report_best_tour(best_tour)
    
    ops = 0
    improved = True
    stagnation = 0
    threshold = 1
    while ops < budget:
        if not improved:
            stagnation += 1
            # Adaptive threshold: increases with budget consumption
            threshold = 1 + int((ops / budget) * 5) if budget > 0 else 6
            if stagnation > threshold:
                # Double-bridge perturbation
                tour = double_bridge(tour)
                cur_dist = compute_dist(tour)
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                # Reset stagnation after perturbation
                stagnation = 0
                improved = True
                continue
        else:
            stagnation = 0
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
                if new < old - 1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    cur_dist = best_dist + (new - old)  # delta update for current distance
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break
            if improved:
                break
    return best_tour