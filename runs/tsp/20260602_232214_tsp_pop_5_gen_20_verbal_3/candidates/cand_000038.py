import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def tour_dist(tour):
        total = 0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i+1)%n]]
        return total

    def regret_construction():
        tour = [0]
        first = np.argmin(distance_matrix[0, 1:]) + 1
        tour.append(first)
        unvisited = set(range(n)) - set(tour)
        while unvisited:
            best_insert = {}
            second_best = {}
            for city in unvisited:
                best = float('inf')
                sec = float('inf')
                best_pos = None
                for pos in range(len(tour)):
                    i = tour[pos]
                    j = tour[(pos+1)%len(tour)]
                    cost = distance_matrix[i, city] + distance_matrix[city, j] - distance_matrix[i, j]
                    if cost < best:
                        sec = best
                        best = cost
                        best_pos = pos
                    elif cost < sec:
                        sec = cost
                best_insert[city] = (best_pos, best)
                second_best[city] = sec if sec != float('inf') else best
            regret = {c: second_best[c] - best_insert[c][1] for c in unvisited}
            chosen = max(unvisited, key=lambda c: (regret[c], -best_insert[c][1]))
            pos, _ = best_insert[chosen]
            tour.insert(pos+1, chosen)
            unvisited.remove(chosen)
        return np.array(tour)

    def two_opt(tour):
        best_tour = tour.copy()
        best_dist = tour_dist(best_tour)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    a = best_tour[i]
                    b = best_tour[(i+1)%n]
                    c = best_tour[j]
                    d = best_tour[(j+1)%n]
                    old = distance_matrix[a, b] + distance_matrix[c, d]
                    new = distance_matrix[a, c] + distance_matrix[b, d]
                    if new < old:
                        new_tour = np.concatenate([best_tour[:i+1], best_tour[i+1:j+1][::-1], best_tour[j+1:]])
                        new_dist = tour_dist(new_tour)
                        if new_dist < best_dist:
                            best_tour = new_tour
                            best_dist = new_dist
                            improved = True
                            report_best_tour(best_tour)
        return best_tour, best_dist

    def double_bridge(tour):
        pos = sorted(np.random.choice(range(1, n), 3, replace=False))
        p1, p2, p3 = pos
        seg1 = tour[:p1]
        seg2 = tour[p1:p2]
        seg3 = tour[p2:p3]
        seg4 = tour[p3:]
        # reconnect as seg1, seg3, seg2, seg4 (or another permutation)
        # choose random permutation that changes order
        perm = np.random.choice([0,1,2,3], 4, replace=False)
        # but to ensure validity, just use a fixed pattern
        new_tour = np.concatenate([seg1, seg3, seg2, seg4])
        # ensure start node remains 0? not necessary, but keep as is
        return new_tour

    # Main
    best_overall_dist = float('inf')
    best_overall_tour = None
    for _ in range(10):  # number of restarts
        tour = regret_construction()
        report_best_tour(tour)
        tour, dist = two_opt(tour)
        if dist < best_overall_dist:
            best_overall_dist = dist
            best_overall_tour = tour
        # perturbation
        if n > 10:
            tour = double_bridge(tour)
            tour, dist = two_opt(tour)
            if dist < best_overall_dist:
                best_overall_dist = dist
                best_overall_tour = tour
    return best_overall_tour