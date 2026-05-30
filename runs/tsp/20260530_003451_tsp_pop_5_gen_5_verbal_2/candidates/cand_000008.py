import numpy as np
def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n == 1:
        tour = np.array([0])
        report_best_tour(tour)
        return tour
    # farthest insertion construction
    tour = [0]
    unvisited = set(range(1, n))
    while unvisited:
        # select farthest unvisited node
        best_node = None
        best_min_dist = -1
        for v in unvisited:
            min_dist = min(distance_matrix[v, t] for t in tour)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_node = v
        # find best insertion position
        best_increase = np.inf
        best_pos = 0
        for i in range(len(tour)):
            prev = tour[i]
            nxt = tour[(i+1) % len(tour)]
            increase = distance_matrix[prev, best_node] + distance_matrix[best_node, nxt] - distance_matrix[prev, nxt]
            if increase < best_increase:
                best_increase = increase
                best_pos = i+1
        tour.insert(best_pos, best_node)
        unvisited.remove(best_node)
    tour = np.array(tour, dtype=np.int32)
    best_tour = tour.copy()
    best_cost = 0.0
    for i in range(n):
        best_cost += distance_matrix[tour[i], tour[(i+1)%n]]
    report_best_tour(best_tour)
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+2, n):
                a, b = tour[i], tour[(i+1)%n]
                c, d = tour[j], tour[(j+1)%n]
                if distance_matrix[a, c] + distance_matrix[b, d] < distance_matrix[a, b] + distance_matrix[c, d]:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    cost = 0.0
                    for k in range(n):
                        cost += distance_matrix[tour[k], tour[(k+1)%n]]
                    if cost < best_cost:
                        best_cost = cost
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    improved = True
                    break
            if improved:
                break
    return best_tour