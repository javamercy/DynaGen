import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n)
    # Farthest insertion construction
    def farthest_insertion():
        max_dist = -1
        start = 0
        end = 1
        for i in range(n):
            for j in range(i+1, n):
                if distance_matrix[i, j] > max_dist:
                    max_dist = distance_matrix[i, j]
                    start, end = i, j
        tour = [start, end]
        in_tour = {start, end}
        while len(tour) < n:
            farthest_city = None
            max_min_dist = -1
            for city in range(n):
                if city in in_tour:
                    continue
                min_dist = min(distance_matrix[city, tour[i]] for i in range(len(tour)))
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    farthest_city = city
            best_pos = 0
            best_increase = float('inf')
            for pos in range(len(tour)):
                prev = tour[pos]
                nxt = tour[(pos+1) % len(tour)]
                increase = distance_matrix[prev, farthest_city] + distance_matrix[farthest_city, nxt] - distance_matrix[prev, nxt]
                if increase < best_increase:
                    best_increase = increase
                    best_pos = pos+1
            tour.insert(best_pos, farthest_city)
            in_tour.add(farthest_city)
        return np.array(tour)
    
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
                        break
                if improved:
                    break
        return tour
    
    def double_bridge(tour):
        # Randomly cut tour into 4 segments and rejoin in a different order
        idx = sorted(np.random.choice(n, 4, replace=False))
        a,b,c,d = idx[0], idx[1], idx[2], idx[3]
        # Segments: [a:b], [b:c], [c:d], [d:]
        # New order: [a:b], [c:d], [b:c], [d:] reversed? Actually standard double bridge: swap two pairs
        tour = np.concatenate([tour[:a], tour[b:c], tour[d:], tour[a:b], tour[c:d]])
        return tour
    
    best_tour = farthest_insertion()
    best_dist = None
    def tour_dist(t):
        return distance_matrix[t[:-1], t[1:]].sum() + distance_matrix[t[-1], t[0]]
    best_dist = tour_dist(best_tour)
    report_best_tour(best_tour)
    best_tour = two_opt(best_tour)
    cur_dist = tour_dist(best_tour)
    if cur_dist < best_dist:
        best_dist = cur_dist
        report_best_tour(best_tour)
    # Iterated local search
    for _ in range(10):
        perturbed = double_bridge(best_tour.copy())
        improved = two_opt(perturbed)
        new_dist = tour_dist(improved)
        if new_dist < best_dist:
            best_dist = new_dist
            best_tour = improved
            report_best_tour(best_tour)
    return best_tour