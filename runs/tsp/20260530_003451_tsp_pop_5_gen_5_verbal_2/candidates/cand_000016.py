import numpy as np
def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    best_cost = np.inf
    best_tour = None
    num_restarts = 10
    for restart in range(num_restarts):
        if restart == 0:
            # nearest neighbor construction
            tour = [0]
            unvisited = set(range(1, n))
            curr = 0
            while unvisited:
                nxt = min(unvisited, key=lambda c: distance_matrix[curr, c])
                tour.append(nxt)
                unvisited.remove(nxt)
                curr = nxt
            tour = np.array(tour, dtype=np.int32)
        else:
            # random permutation
            tour = np.random.permutation(n).astype(np.int32)
        # local search: full 2-opt until no improvement
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i + 2, n):
                    a, b = tour[i], tour[(i + 1) % n]
                    c, d = tour[j], tour[(j + 1) % n]
                    if distance_matrix[a, c] + distance_matrix[b, d] < distance_matrix[a, b] + distance_matrix[c, d]:
                        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                        improved = True
                        break
                if improved:
                    break
        # compute cost
        cost = 0.0
        for k in range(n):
            cost += distance_matrix[tour[k], tour[(k + 1) % n]]
        if cost < best_cost:
            best_cost = cost
            best_tour = tour.copy()
            report_best_tour(best_tour)
    return best_tour