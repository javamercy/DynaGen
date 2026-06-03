import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0])
        report_best_tour(tour)
        return tour
    if n == 2:
        tour = np.array([0, 1])
        report_best_tour(tour)
        return tour

    def total_dist(t):
        return distance_matrix[t[-1], t[0]] + np.sum(distance_matrix[t[:-1], t[1:]])

    def steepest_two_opt(tour):
        improved = True
        while improved:
            improved = False
            best_delta = 0
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
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            if best_delta < -1e-12:
                i, j = best_i, best_j
                tour[i+1:j+1] = tour[i+1:j+1][::-1]
                improved = True
        return tour

    def farthest_insertion(start):
        tour = [start]
        visited = {start}
        unvisited = set(range(n)) - visited
        while unvisited:
            # find farthest unvisited node from tour
            farthest_node = None
            farthest_dist = -1
            for v in unvisited:
                min_dist = min(distance_matrix[v, t] for t in tour)
                if min_dist > farthest_dist:
                    farthest_dist = min_dist
                    farthest_node = v
            # insert at best position
            best_gain = np.inf
            best_pos = 0
            for pos in range(len(tour)):
                next_pos = (pos + 1) % len(tour)
                gain = (distance_matrix[tour[pos], farthest_node] +
                        distance_matrix[farthest_node, tour[next_pos]] -
                        distance_matrix[tour[pos], tour[next_pos]])
                if gain < best_gain:
                    best_gain = gain
                    best_pos = pos + 1
            tour.insert(best_pos, farthest_node)
            visited.add(farthest_node)
            unvisited.remove(farthest_node)
        return tour

    best_tour = None
    best_dist = np.inf
    for _ in range(10):
        start = np.random.randint(n)
        tour = farthest_insertion(start)
        tour = steepest_two_opt(tour)
        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = np.array(tour)
            report_best_tour(best_tour)
    return best_tour