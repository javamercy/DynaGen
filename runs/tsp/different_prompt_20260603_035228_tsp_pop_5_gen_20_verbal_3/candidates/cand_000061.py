import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def total_dist(tour):
        return distance_matrix[tour[-1], tour[0]] + np.sum(distance_matrix[tour[:-1], tour[1:]])

    def nearest_neighbor(start):
        tour = [start]
        visited = {start}
        for _ in range(n-1):
            last = tour[-1]
            best = -1
            bestd = np.inf
            for j in range(n):
                if j not in visited and distance_matrix[last, j] < bestd:
                    bestd = distance_matrix[last, j]
                    best = j
            tour.append(best)
            visited.add(best)
        return tour

    def two_opt_steepest(tour):
        improved = True
        while improved:
            improved = False
            best_gain = 0.0
            best_i = best_j = None
            for i in range(n-1):
                for j in range(i+2, n):
                    a = tour[i]; b = tour[i+1]; c = tour[j]; d = tour[(j+1)%n]
                    current = distance_matrix[a,b] + distance_matrix[c,d]
                    proposed = distance_matrix[a,c] + distance_matrix[b,d]
                    gain = current - proposed
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_i, best_j = i, j
            if best_gain > 1e-12:
                i, j = best_i, best_j
                tour[i+1:j+1] = reversed(tour[i+1:j+1])
                improved = True
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
    num_restarts = 10
    max_cycles = 30
    stall_limit = 5

    for _ in range(num_restarts):
        start = np.random.randint(n)
        tour = nearest_neighbor(start)
        tour = two_opt_steepest(tour)
        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = np.array(tour)
            report_best_tour(best_tour)

        no_improve = 0
        for cycle in range(max_cycles):
            tour = two_opt_steepest(tour)
            cur_dist = total_dist(tour)
            if cur_dist < best_dist - 1e-12:
                best_dist = cur_dist
                best_tour = np.array(tour)
                report_best_tour(best_tour)
                no_improve = 0
            else:
                no_improve += 1

            if cycle < max_cycles - 1:
                if no_improve >= stall_limit:
                    tour = double_bridge(tour)
                    no_improve = 0
                else:
                    seg_len = np.random.randint(2, min(n//4 + 2, n))
                    i = np.random.randint(0, n - seg_len)
                    tour[i:i+seg_len] = reversed(tour[i:i+seg_len])

    return best_tour