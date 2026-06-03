import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def total_dist(t):
        idx = np.array(t)
        return distance_matrix[idx[-1], idx[0]] + np.sum(distance_matrix[idx[:-1], idx[1:]])

    def farthest_insertion():
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            farthest_city = None
            farthest_dist = -1
            for city in unvisited:
                min_dist = min(distance_matrix[city, t] for t in tour)
                if min_dist > farthest_dist:
                    farthest_dist = min_dist
                    farthest_city = city
            best_increase = float('inf')
            best_pos = 0
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                increase = (distance_matrix[tour[i], farthest_city] +
                            distance_matrix[farthest_city, tour[j]] -
                            distance_matrix[tour[i], tour[j]])
                if increase < best_increase:
                    best_increase = increase
                    best_pos = j
            tour.insert(best_pos, farthest_city)
            unvisited.remove(farthest_city)
        return tour

    def two_opt_first(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n-1):
                for j in range(i+2, n):
                    if j == n-1:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[0]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[0]])
                    else:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[j+1]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[j+1]])
                    if delta > 1e-12:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
                        break
                if improved:
                    break
        return tour

    def double_bridge(tour):
        cuts = sorted(np.random.choice(range(1, n), 3, replace=False))
        seg0 = tour[:cuts[0]]
        seg1 = tour[cuts[0]:cuts[1]]
        seg2 = tour[cuts[1]:cuts[2]]
        seg3 = tour[cuts[2]:]
        return seg0 + seg2 + seg1 + seg3

    best_tour = None
    best_dist = float('inf')
    num_restarts = 20
    max_iter = 50

    for _ in range(num_restarts):
        tour = farthest_insertion()
        cur_dist = total_dist(tour)
        if cur_dist < best_dist:
            best_dist = cur_dist
            best_tour = np.array(tour)
            report_best_tour(best_tour)

        no_improve = 0
        for _ in range(max_iter):
            tour = two_opt_first(tour)
            cur_dist = total_dist(tour)
            if cur_dist < best_dist - 1e-12:
                best_dist = cur_dist
                best_tour = np.array(tour)
                report_best_tour(best_tour)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= 3:
                tour = double_bridge(tour)
                no_improve = 0

    return best_tour