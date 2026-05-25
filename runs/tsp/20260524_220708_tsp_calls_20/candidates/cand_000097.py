import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        np.random.default_rng(seed).shuffle(tour)
        report_best_tour(tour)
        return tour
    
    rng = np.random.default_rng(seed)
    dist = distance_matrix
    
    def random_insertion_construction():
        start = rng.integers(n)
        tour = [start]
        unvisited = list(range(n))
        unvisited.remove(start)
        rng.shuffle(unvisited)
        for city in unvisited:
            best_inc = np.inf
            best_pos = 0
            m = len(tour)
            for i in range(m):
                prev = tour[i]
                nxt = tour[(i+1) % m]
                inc = dist[prev, city] + dist[city, nxt] - dist[prev, nxt]
                if inc < best_inc:
                    best_inc = inc
                    best_pos = i+1
            tour.insert(best_pos, city)
        return np.array(tour, dtype=np.int64)
    
    def compute_dist(tour):
        d = 0.0
        for i in range(n):
            d += dist[tour[i], tour[(i+1) % n]]
        return d
    
    tour = random_insertion_construction()
    cur_dist = compute_dist(tour)
    best_tour = tour.copy()
    best_dist = cur_dist
    report_best_tour(best_tour)
    
    ops = 0
    no_improve_passes = 0
    threshold = max(3, n // 10)
    
    while ops < budget:
        # First-improvement 2-opt pass
        improved = False
        for i in range(n-1):
            if ops >= budget:
                break
            for j in range(i+2, n):
                if ops >= budget:
                    break
                ops += 1
                a, b = tour[i], tour[(i+1) % n]
                c, d = tour[j], tour[(j+1) % n]
                old = dist[a, b] + dist[c, d]
                new = dist[a, c] + dist[b, d]
                if new < old - 1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    cur_dist += new - old
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = tour.copy()
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
            if no_improve_passes >= threshold or (rng.random() < 0.15 and no_improve_passes > 0):
                if ops >= budget:
                    break
                ops += 1  # count perturbation as one operation
                if rng.random() < 0.5:
                    # Large segment removal
                    L = max(2, n // 4)
                    start_idx = rng.integers(0, n - L + 1)
                    segment = set(tour[start_idx:start_idx+L])
                    base = [c for c in tour if c not in segment]
                    for city in segment:
                        best_inc = np.inf
                        best_pos = 0
                        m = len(base)
                        for i in range(m):
                            prev = base[i]
                            nxt = base[(i+1) % m]
                            inc = dist[prev, city] + dist[city, nxt] - dist[prev, nxt]
                            if inc < best_inc:
                                best_inc = inc
                                best_pos = i+1
                        base.insert(best_pos, city)
                    new_tour = np.array(base, dtype=np.int64)
                    new_dist = compute_dist(new_tour)
                else:
                    # Double-bridge move
                    # Split into 4 segments
                    p = [0]
                    for _ in range(3):
                        p.append(rng.integers(p[-1]+1, n - (3 - _)))
                    p.append(n)
                    # reorder: 0-3-2-1-4
                    segments = [tour[p[0]:p[1]], tour[p[1]:p[2]], tour[p[2]:p[3]], tour[p[3]:p[4]]]
                    new_tour = np.concatenate([segments[0], segments[3], segments[2], segments[1]], dtype=np.int64)
                    new_dist = compute_dist(new_tour)
                
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_tour = new_tour.copy()
                    report_best_tour(best_tour)
                # Acceptance
                if new_dist <= best_dist * 1.05:
                    tour = new_tour
                    cur_dist = new_dist
                else:
                    accept_prob = np.exp(-(new_dist - cur_dist) / max(1e-6, best_dist * 0.1))
                    if rng.random() < accept_prob:
                        tour = new_tour
                        cur_dist = new_dist
                no_improve_passes = 0
            
            if no_improve_passes >= 2 * threshold:
                if ops >= budget:
                    break
                # Restart from new random insertion construction
                tour = random_insertion_construction()
                cur_dist = compute_dist(tour)
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                no_improve_passes = 0
    
    return best_tour