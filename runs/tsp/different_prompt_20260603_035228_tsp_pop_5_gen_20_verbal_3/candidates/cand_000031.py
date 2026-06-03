import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def total_dist(t):
        if len(t) <= 1:
            return 0.0
        return distance_matrix[t[-1], t[0]] + np.sum(distance_matrix[t[:-1], t[1:]])

    def steepest_2opt(tour):
        improved = True
        while improved:
            improved = False
            best_gain = 0.0
            best_i = best_j = None
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == n - 1:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[0]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[0]])
                    else:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[j+1]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[j+1]])
                    if delta > best_gain + 1e-12:
                        best_gain = delta
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

    def node_insertion(tour):
        best_tour = tour[:]
        best_dist = total_dist(best_tour)
        for i in range(n):
            node = best_tour.pop(i)
            for j in range(n):
                if j == i:
                    continue
                new_tour = best_tour[:j] + [node] + best_tour[j:]
                d = total_dist(new_tour)
                if d < best_dist - 1e-12:
                    best_dist = d
                    best_tour = new_tour
                    # adjust i for next iteration? We'll recompute later
                    # break to avoid complexity; we'll do full pass
            
        return best_tour

    best_tour = None
    best_dist = float('inf')
    num_restarts = 3
    max_cycles = 30
    stall_limit = 3

    for _ in range(num_restarts):
        # Nearest neighbor with random start
        start = np.random.randint(n)
        tour = [start]
        visited = [False] * n
        visited[start] = True
        for _ in range(n - 1):
            last = tour[-1]
            best = -1
            bestd = float('inf')
            for j in range(n):
                if not visited[j] and distance_matrix[last, j] < bestd:
                    bestd = distance_matrix[last, j]
                    best = j
            tour.append(best)
            visited[best] = True

        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = np.array(tour)
            report_best_tour(best_tour)

        no_improve = 0
        for cycle in range(max_cycles):
            tour = steepest_2opt(tour)
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
                    # apply node insertion perturbation
                    tour = node_insertion(tour)
                    cur_dist = total_dist(tour)
                    if cur_dist < best_dist - 1e-12:
                        best_dist = cur_dist
                        best_tour = np.array(tour)
                        report_best_tour(best_tour)
                    no_improve = 0
                else:
                    # random segment reversal of length 2 to n//4
                    seg_len = np.random.randint(2, max(3, n//4 + 2))
                    i = np.random.randint(0, n - seg_len)
                    tour[i:i+seg_len] = reversed(tour[i:i+seg_len])
                    # sometimes double bridge
                    if np.random.rand() < 0.5:
                        tour = double_bridge(tour)

    return best_tour