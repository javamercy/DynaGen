import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    
    best_tour = None
    best_dist = np.inf
    
    for restart in range(50):
        # 2-regret insertion
        start = random.sample(range(n), 3)
        tour = list(start)
        remaining = set(range(n)) - set(start)
        while remaining:
            best_city = -1
            best_regret = -1.0
            best_pos = -1
            best_cost = np.inf
            for city in remaining:
                costs = []
                for pos in range(len(tour)):
                    before = tour[pos-1]
                    after = tour[pos] if pos < len(tour) else tour[0]
                    delta = distance_matrix[before, city] + distance_matrix[city, after] - distance_matrix[before, after]
                    costs.append((delta, pos))
                costs.sort(key=lambda x: x[0])
                first = costs[0][0]
                second = costs[1][0] if len(costs) > 1 else first
                regret = second - first
                if regret > best_regret or (regret == best_regret and city < best_city):
                    best_regret = regret
                    best_city = city
                    best_pos = costs[0][1]
                    best_cost = first
            tour.insert(best_pos, best_city)
            remaining.remove(best_city)
        
        tour_arr = np.array(tour, dtype=np.int64)
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour_arr[i], tour_arr[(i+1)%n]]
        
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = tour_arr.copy()
            report_best_tour(best_tour)
        
        # VND + perturbation
        improved = True
        while improved:
            improved = False
            # Or-opt
            for L in [1,2,3]:
                if L >= n:
                    continue
                start_i = 0
                while start_i < n:
                    seg_indices = [(start_i + k) % n for k in range(L)]
                    seg = [tour[idx] for idx in seg_indices]
                    before_rem = tour[(seg_indices[0]-1)%n]
                    after_rem = tour[(seg_indices[-1]+1)%n]
                    removed_edges = distance_matrix[before_rem, tour[seg_indices[0]]] + distance_matrix[tour[seg_indices[-1]], after_rem]
                    added_edge = distance_matrix[before_rem, after_rem]
                    delta_rem = added_edge - removed_edges
                    new_tour_no_seg = [tour[j] for j in range(n) if j not in seg_indices]
                    m = n - L
                    found = False
                    for pos in range(m+1):
                        if pos == 0:
                            pre = new_tour_no_seg[-1]
                            post = new_tour_no_seg[0]
                        elif pos == m:
                            pre = new_tour_no_seg[-1]
                            post = new_tour_no_seg[0]
                        else:
                            pre = new_tour_no_seg[pos-1]
                            post = new_tour_no_seg[pos]
                        old_edge = distance_matrix[pre, post]
                        new_edges = distance_matrix[pre, seg[0]] + distance_matrix[seg[-1], post]
                        delta_ins = new_edges - old_edge
                        total_delta = delta_rem + delta_ins
                        if total_delta < -1e-10:
                            candidate = new_tour_no_seg[:pos] + seg + new_tour_no_seg[pos:]
                            dist += total_delta
                            tour = candidate
                            if dist < best_dist - 1e-10:
                                best_dist = dist
                                best_tour = np.array(tour)
                                report_best_tour(best_tour)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                    start_i += 1
                if improved:
                    break
            if improved:
                continue
            # 2-opt
            for i in range(n):
                for j in range(i+2, n):
                    a, b, c, d = tour[i], tour[(i+1)%n], tour[j], tour[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-10:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        dist += delta
                        if dist < best_dist - 1e-10:
                            best_dist = dist
                            best_tour = np.array(tour)
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved:
                    break
        
        # Double-bridge perturbation
        u = random.randint(0, n-1)
        v = random.randint(u+2, n-2)
        w = random.randint(v+2, n-1)
        x = random.randint(w+2, n-1)
        # Ensure four distinct segments
        if u >= v or v >= w or w >= x:
            continue
        # Extract segments: A=[0..u], B=[u+1..v], C=[v+1..w], D=[w+1..x], E=[x+1..n-1]
        A = tour[:u+1]
        B = tour[u+1:v+1]
        C = tour[v+1:w+1]
        D = tour[w+1:x+1]
        E = tour[x+1:]
        # Reorder: A, D, B, E, C (or any other permutation)
        new_tour = A + D + B + E + C
        # Evaluate new distance
        new_dist = 0.0
        for i in range(n):
            new_dist += distance_matrix[new_tour[i], new_tour[(i+1)%n]]
        tour = new_tour
        dist = new_dist
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = np.array(tour)
            report_best_tour(best_tour)
        # Accept the perturbation unconditionally (simulated annealing-like, but here we always accept)
    
    return best_tour