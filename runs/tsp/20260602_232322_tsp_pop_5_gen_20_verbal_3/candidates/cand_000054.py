import numpy as np
import random

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    
    best_tour = None
    best_dist = float('inf')
    restarts = min(20, n)
    
    for _ in range(restarts):
        # Cheapest insertion construction
        start_city = random.randrange(n)
        tour = [start_city]
        remaining = set(range(n)) - {start_city}
        while remaining:
            best_delta = float('inf')
            best_cities = []
            for city in remaining:
                # compute delta for best insertion position
                for pos in range(len(tour)):
                    before = tour[pos-1]
                    after = tour[pos] if pos < len(tour) else tour[0]
                    delta = distance_matrix[before, city] + distance_matrix[city, after] - distance_matrix[before, after]
                    if delta < best_delta - 1e-10:
                        best_delta = delta
                        best_cities = [(city, pos)]
                    elif abs(delta - best_delta) < 1e-10:
                        best_cities.append((city, pos))
            # Random tie-break among best candidates
            chosen, pos = random.choice(best_cities)
            tour.insert(pos, chosen)
            remaining.remove(chosen)
        
        tour = np.array(tour, dtype=int)
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour[i], tour[(i+1)%n]]
        
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        
        # Local search with perturbation (ILS)
        for _ in range(3):  # at most 3 perturbations
            improved = True
            while improved:
                improved = False
                # 2-opt first improvement
                for i in range(n-1):
                    for j in range(i+2, n):
                        a = tour[i]
                        b = tour[(i+1)%n]
                        c = tour[j]
                        d = tour[(j+1)%n]
                        delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                        if delta < -1e-10:
                            new_tour = tour.copy()
                            new_tour[i+1:j+1] = tour[j:i:-1]
                            tour = new_tour
                            dist += delta
                            if dist < best_dist - 1e-10:
                                best_dist = dist
                                best_tour = tour.copy()
                                report_best_tour(best_tour)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    continue
                # Or-opt with L=1,2,3
                for L in (1,2,3):
                    if L >= n:
                        continue
                    for i in range(n):
                        # Extract segment of length L starting at i
                        if i + L <= n:
                            seg = list(tour[i:i+L])
                            new_tour_list = list(tour[:i]) + list(tour[i+L:])
                        else:
                            # wrap around
                            seg = list(tour[i:]) + list(tour[:i+L-n])
                            new_tour_list = list(tour[i+L-n:i])
                        m = len(new_tour_list)
                        if m == 0:
                            continue
                        for pos in range(m+1):
                            cand = new_tour_list[:pos] + seg + new_tour_list[pos:]
                            cand_dist = 0.0
                            for k in range(n):
                                cand_dist += distance_matrix[cand[k], cand[(k+1)%n]]
                            if cand_dist < dist - 1e-10:
                                tour = np.array(cand, dtype=int)
                                dist = cand_dist
                                if dist < best_dist - 1e-10:
                                    best_dist = dist
                                    best_tour = tour.copy()
                                    report_best_tour(best_tour)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            # Perturbation: random reversal of a segment
            if improved is False:  # local optimum reached
                i = random.randrange(n)
                j = random.randrange(n)
                if i > j:
                    i, j = j, i
                # reverse segment from i to j (inclusive-? we use exclusive end like 2-opt)
                # reverse segment from i to j (inclusive of i, exclusive of j+1?)
                # we'll do simple: take segment and reverse it, place back
                seg_len = j - i + 1
                if seg_len < 2:
                    continue
                # Apply perturbation only if it does not worsen too much? We just do it.
                new_tour = tour.copy()
                new_tour[i:j+1] = tour[j:i-1:-1] if i > 0 else np.concatenate([tour[j::-1], tour[:i-1:-1]])? 
                # Actually easier: convert to list, reverse slice, convert back
                tour_list = tour.tolist()
                tour_list[i:j+1] = reversed(tour_list[i:j+1])
                tour = np.array(tour_list, dtype=int)
                dist = 0.0
                for k in range(n):
                    dist += distance_matrix[tour[k], tour[(k+1)%n]]
                # no need to report unless improved, but we will re-run local search
    return best_tour