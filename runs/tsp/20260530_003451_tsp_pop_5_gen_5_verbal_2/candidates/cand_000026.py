import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n)

    def tour_distance(tour):
        return sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))

    def nearest_insertion():
        # start with two farthest cities
        max_dist = -1
        start, end = 0, 1
        for i in range(n):
            for j in range(i+1, n):
                d = distance_matrix[i, j]
                if d > max_dist:
                    max_dist = d
                    start, end = i, j
        tour = [start, end]
        in_tour = {start, end}
        # randomized farthest insertion with squared distances
        while len(tour) < n:
            candidates = []
            weights = []
            for city in range(n):
                if city in in_tour:
                    continue
                min_dist = min(distance_matrix[city, t] for t in tour)
                candidates.append(city)
                weights.append(min_dist * min_dist + 1e-6)
            total = sum(weights)
            r = random.random() * total
            cum = 0
            for idx, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen = candidates[idx]
                    break
            # best insertion position
            best_inc = float('inf')
            best_pos = 0
            for pos in range(len(tour)):
                prev = tour[pos]
                nxt = tour[(pos+1)%len(tour)]
                inc = distance_matrix[prev, chosen] + distance_matrix[chosen, nxt] - distance_matrix[prev, nxt]
                if inc < best_inc:
                    best_inc = inc
                    best_pos = pos+1
            tour.insert(best_pos, chosen)
            in_tour.add(chosen)
        return tour

    def two_opt(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                    if distance_matrix[a, c] + distance_matrix[b, d] < distance_matrix[a, b] + distance_matrix[c, d]:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
        return tour

    def double_bridge(tour):
        # random 4 breakpoints
        p = sorted(random.sample(range(n), 4))
        a, b, c, d = p[0], p[1], p[2], p[3]
        # segments
        seg1 = tour[0:a]
        seg2 = tour[a:b]
        seg3 = tour[b:c]
        seg4 = tour[c:d]
        seg5 = tour[d:]
        # reorder: seg1, seg3, seg2, seg4, seg5 (double-bridge typical)
        new_tour = seg1 + seg3 + seg2 + seg4 + seg5
        # Ensure length n
        if len(new_tour) != n:
            # fallback to original
            return tour[:]
        return new_tour

    best_tour = None
    best_dist = float('inf')

    for restart in range(5):
        # construction
        tour = nearest_insertion()
        # 2-opt
        tour = two_opt(tour)
        current_dist = tour_distance(tour)
        if current_dist < best_dist:
            best_dist = current_dist
            best_tour = tour[:]
            report_best_tour(np.array(best_tour))
        # iterated local search
        for _ in range(10):
            new_tour = double_bridge(tour)
            new_tour = two_opt(new_tour)
            new_dist = tour_distance(new_tour)
            if new_dist < current_dist:
                tour = new_tour
                current_dist = new_dist
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_tour = new_tour[:]
                    report_best_tour(np.array(best_tour))
            else:
                # maybe accept with probability? skip for simplicity keep hill climbing
                pass

    return np.array(best_tour)