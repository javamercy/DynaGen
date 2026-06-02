import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n, dtype=int)
        report_best_tour(tour)
        return tour

    best_tour = None
    best_cost = float('inf')

    for restart in range(20):  # number of restarts
        tour = np.random.permutation(n).astype(int)
        cost = _tour_cost(distance_matrix, tour)
        improved = True
        while improved:
            improved = False
            # 2-opt
            for i in range(n):
                for j in range(i+2, n):
                    if j == i+1:
                        continue
                    # compute delta for 2-opt swap (reverse segment i..j-1)
                    i_prev = (i-1) % n
                    j_cur = j % n
                    delta = (distance_matrix[tour[i_prev], tour[j_cur]] +
                             distance_matrix[tour[i], tour[(j+1)%n]] -
                             distance_matrix[tour[i_prev], tour[i]] -
                             distance_matrix[tour[j_cur], tour[(j+1)%n]])
                    if delta < -1e-12:
                        # reverse segment
                        tour[i:j] = tour[i:j][::-1]
                        cost += delta
                        improved = True
                        if cost < best_cost:
                            best_cost = cost
                            best_tour = tour.copy()
                            report_best_tour(best_tour)
            # Or-opt: move segments of length L = 1,2,3
            for L in [1, 2, 3]:
                for i in range(n):
                    j = (i + L) % n
                    # segment from i to j-1 (inclusive) to be moved
                    # we'll try insert after position k (not inside segment)
                    for k in range(n):
                        if (k >= i-1 and k <= j) or (k+L >= i and k+L <= j):
                            continue
                        # compute delta for removing segment and inserting after k
                        # current edges: (i-1, i), (j-1, j), (k, k+1)
                        # after: (i-1, j), (k, i), (j-1, k+1)
                        i_prev = (i-1) % n
                        j_minus1 = (j-1) % n
                        k_next = (k+1) % n
                        delta = (distance_matrix[tour[i_prev], tour[j]] +
                                 distance_matrix[tour[k], tour[i]] +
                                 distance_matrix[tour[j_minus1], tour[k_next]] -
                                 distance_matrix[tour[i_prev], tour[i]] -
                                 distance_matrix[tour[j_minus1], tour[j]] -
                                 distance_matrix[tour[k], tour[k_next]])
                        if delta < -1e-12:
                            # apply move
                            segment = tour[i:j].copy()
                            if k < i:
                                tour = np.concatenate([tour[:k+1], segment, tour[k+1:i], tour[j:]])
                            elif k >= j:
                                tour = np.concatenate([tour[:i], tour[j:k+1], segment, tour[k+1:]])
                            else:
                                # shouldn't happen
                                continue
                            cost += delta
                            improved = True
                            if cost < best_cost:
                                best_cost = cost
                                best_tour = tour.copy()
                                report_best_tour(best_tour)
        # after VND, update best if needed
        if cost < best_cost:
            best_cost = cost
            best_tour = tour.copy()
            report_best_tour(best_tour)
    return best_tour

def _tour_cost(dm, tour):
    n = len(tour)
    cost = dm[tour[-1], tour[0]]
    for k in range(n-1):
        cost += dm[tour[k], tour[k+1]]
    return cost