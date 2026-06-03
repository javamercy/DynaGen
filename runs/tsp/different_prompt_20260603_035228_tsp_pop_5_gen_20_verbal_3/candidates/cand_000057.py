import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n, dtype=int)
        report_best_tour(tour)
        return tour

    def total_dist(tour):
        return distance_matrix[tour[-1], tour[0]] + np.sum(distance_matrix[tour[:-1], tour[1:]])

    def farthest_insertion():
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n)) - {start}
        while unvisited:
            farthest = max(unvisited, key=lambda x: min(distance_matrix[x, t] for t in tour))
            best_pos = 0
            best_cost = float('inf')
            for pos in range(len(tour) + 1):
                if pos == 0:
                    cost = distance_matrix[tour[-1], farthest] + distance_matrix[farthest, tour[0]] - distance_matrix[tour[-1], tour[0]]
                elif pos == len(tour):
                    cost = distance_matrix[tour[-1], farthest] + distance_matrix[farthest, tour[0]] - distance_matrix[tour[-1], tour[0]]
                else:
                    cost = distance_matrix[tour[pos-1], farthest] + distance_matrix[farthest, tour[pos]] - distance_matrix[tour[pos-1], tour[pos]]
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
            tour.insert(best_pos, farthest)
            unvisited.remove(farthest)
        return np.array(tour, dtype=int)

    def two_opt(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    a, b = tour[i], tour[(i + 1) % n]
                    c, d = tour[j], tour[(j + 1) % n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-9:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
        return tour

    def double_bridge(tour):
        cuts = sorted(np.random.choice(range(1, n), 3, replace=False))
        return np.concatenate([tour[:cuts[0]], tour[cuts[1]:cuts[2]], tour[cuts[0]:cuts[1]], tour[cuts[2]:]])

    def node_insertion(tour):
        i, j = np.random.choice(n, 2, replace=False)
        if i > j:
            i, j = j, i
        node = tour[j]
        tour = np.delete(tour, j)
        tour = np.insert(tour, i, node)
        return tour

    best_tour = None
    best_dist = float('inf')
    num_restarts = 10
    max_iter = 20

    for _ in range(num_restarts):
        tour = farthest_insertion()
        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-9:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)

        no_improve = 0
        for _ in range(max_iter):
            tour = two_opt(tour)
            cur_dist = total_dist(tour)
            if cur_dist < best_dist - 1e-9:
                best_dist = cur_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= 2:
                if np.random.rand() < 0.7:
                    tour = double_bridge(tour)
                else:
                    tour = node_insertion(tour)
                no_improve = 0

    return best_tour