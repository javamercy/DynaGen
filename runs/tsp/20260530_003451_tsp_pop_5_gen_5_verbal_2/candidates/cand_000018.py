import numpy as np
def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def tour_distance(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))

    def farthest_insertion_construction():
        max_dist = -1
        for i in range(n):
            for j in range(i+1, n):
                d = distance_matrix[i][j]
                if d > max_dist:
                    max_dist = d
                    start, end = i, j
        tour = [start, end]
        in_tour = {start, end}
        while len(tour) < n:
            farthest_city = None
            max_min_dist = -1
            for city in range(n):
                if city in in_tour:
                    continue
                min_dist = min(distance_matrix[city][t] for t in tour)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    farthest_city = city
            best_pos = 0
            best_increase = float('inf')
            for pos in range(len(tour)):
                prev = tour[pos]
                nxt = tour[(pos+1)%len(tour)]
                increase = distance_matrix[prev][farthest_city] + distance_matrix[farthest_city][nxt] - distance_matrix[prev][nxt]
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
                    a, b = tour[i], tour[i+1]
                    c, d = tour[j], tour[(j+1)%n]
                    if distance_matrix[a][c] + distance_matrix[b][d] < distance_matrix[a][b] + distance_matrix[c][d]:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
                        break
                if improved:
                    break
        return tour

    def double_bridge_kick(tour):
        a = np.random.randint(0, n//3)
        b = np.random.randint(a+1, 2*n//3)
        c = np.random.randint(b+1, n-1)
        seg1 = tour[:a]
        seg2 = tour[a:b]
        seg3 = tour[b:c]
        seg4 = tour[c:]
        # reorder: seg2, seg4, seg3, seg1 (or any different order)
        new_tour = np.concatenate([seg2, seg4, seg3, seg1])
        return new_tour

    best_tour = farthest_insertion_construction()
    best_dist = tour_distance(best_tour)
    report_best_tour(best_tour)
    current_tour = best_tour.copy()
    for restart in range(5):
        current_tour = two_opt(current_tour)
        cur_dist = tour_distance(current_tour)
        if cur_dist < best_dist:
            best_dist = cur_dist
            best_tour = current_tour.copy()
            report_best_tour(best_tour)
        # perturbation
        if restart < 4:
            current_tour = double_bridge_kick(current_tour)
            # ensure it's still valid (it is)
    return best_tour