import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)
    if n == 2:
        return np.array([0, 1], dtype=np.int64)

    best_tour = None
    best_cost = np.inf
    n_trials = 10

    for _ in range(n_trials):
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)

        while unvisited:
            best_node = None
            best_cost_inc = np.inf
            best_pos = 0
            for node in unvisited:
                # evaluate insertion cost at each position
                min_inc = np.inf
                min_pos = 0
                for i in range(len(tour)):
                    prev = tour[i]
                    nxt = tour[(i+1) % len(tour)]
                    inc = distance_matrix[prev, node] + distance_matrix[node, nxt] - distance_matrix[prev, nxt]
                    if inc < min_inc:
                        min_inc = inc
                        min_pos = i+1
                if min_inc < best_cost_inc:
                    best_cost_inc = min_inc
                    best_node = node
                    best_pos = min_pos
            tour.insert(best_pos, best_node)
            unvisited.remove(best_node)

        tour = np.array(tour, dtype=np.int64)
        cost = distance_matrix[tour[-1], tour[0]] + sum(distance_matrix[tour[i], tour[i+1]] for i in range(n-1))
        if cost < best_cost:
            best_cost = cost
            best_tour = tour.copy()
            report_best_tour(best_tour)

        # 2-opt improvement
        improved = True
        while improved:
            improved = False
            for i in range(n-1):
                for j in range(i+2, n):
                    if j == i+1:
                        continue
                    a, b, c, d = tour[i], tour[(i+1)%n], tour[j], tour[(j+1)%n]
                    old = distance_matrix[a, b] + distance_matrix[c, d]
                    new = distance_matrix[a, c] + distance_matrix[b, d]
                    if new < old:
                        tour[i+1:j+1] = np.flip(tour[i+1:j+1])
                        improved = True
                        # report new tour if better
                        new_cost = distance_matrix[tour[-1], tour[0]] + sum(distance_matrix[tour[i], tour[i+1]] for i in range(n-1))
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_tour = tour.copy()
                            report_best_tour(best_tour)
                        break
                if improved:
                    break

    return best_tour