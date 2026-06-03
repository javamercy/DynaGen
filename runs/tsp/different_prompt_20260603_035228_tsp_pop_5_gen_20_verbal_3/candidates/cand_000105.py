import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def total_dist(t):
        return np.sum(distance_matrix[t[:-1], t[1:]]) + distance_matrix[t[-1], t[0]]

    def steepest_two_opt(tour):
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
                tour[i+1:j+1] = tour[i+1:j+1][::-1]
                improved = True
        return tour

    def double_bridge(tour):
        cuts = sorted(np.random.choice(range(1, n), 3, replace=False))
        segs = [tour[:cuts[0]], tour[cuts[0]:cuts[1]], tour[cuts[1]:cuts[2]], tour[cuts[2]:]]
        return np.concatenate([segs[0], segs[2], segs[1], segs[3]])

    def cheapest_insertion():
        unvisited = set(range(n))
        start = np.random.randint(n)
        tour = [start]
        unvisited.remove(start)
        nearest = min(unvisited, key=lambda x: distance_matrix[start, x])
        tour.append(nearest)
        unvisited.remove(nearest)
        while unvisited:
            best_increase = float('inf')
            best_node = None
            best_pos = None
            for node in unvisited:
                for pos in range(len(tour)):
                    prev = tour[pos]
                    nxt = tour[(pos+1) % len(tour)]
                    increase = distance_matrix[prev, node] + distance_matrix[node, nxt] - distance_matrix[prev, nxt]
                    if increase < best_increase:
                        best_increase = increase
                        best_node = node
                        best_pos = pos+1
            tour.insert(best_pos, best_node)
            unvisited.remove(best_node)
        return np.array(tour, dtype=int)

    best_tour = None
    best_dist = float('inf')
    num_restarts = 30
    max_cycles = 30
    stall_limit = 5

    for _ in range(num_restarts):
        tour = cheapest_insertion()
        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)

        no_improve = 0
        for cycle in range(max_cycles):
            tour = steepest_two_opt(tour)
            cur_dist = total_dist(tour)
            if cur_dist < best_dist - 1e-12:
                best_dist = cur_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= stall_limit:
                tour = cheapest_insertion()
                no_improve = 0
            else:
                if no_improve < 3:
                    seg_len = np.random.randint(2, max(4, n//3 + 1))
                    i = np.random.randint(0, n - seg_len)
                    tour[i:i+seg_len] = tour[i:i+seg_len][::-1]
                else:
                    tour = double_bridge(tour)

    return best_tour