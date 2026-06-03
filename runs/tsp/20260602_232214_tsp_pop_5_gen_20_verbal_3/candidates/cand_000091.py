import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    # Cheapest Insertion
    start = 0
    tour = [start, (start+1)%n]
    cost = distance_matrix[tour[0], tour[1]] * 2
    in_tour = {tour[0], tour[1]}
    for _ in range(2, n):
        best_inc = np.inf
        best_node = None
        for v in range(n):
            if v in in_tour:
                continue
            # find best insertion position
            for pos in range(len(tour)):
                a = tour[pos]
                b = tour[(pos+1)%len(tour)]
                inc = distance_matrix[a,v] + distance_matrix[v,b] - distance_matrix[a,b]
                if inc < best_inc:
                    best_inc = inc
                    best_node = v
                    best_pos = pos
        tour.insert(best_pos+1, best_node)
        in_tour.add(best_node)
        cost += best_inc
    tour = np.array(tour)
    best_tour = tour.copy()
    best_cost = cost
    report_best_tour(tour)

    # Simulated Annealing
    def two_opt_delta(idx, i, j):
        a,b,c,d = idx[i], idx[i+1], idx[j], idx[(j+1)%n]
        return distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]

    current_cost = cost
    rng = np.random.default_rng()
    T = current_cost * 0.01
    if T == 0: T = 1
    alpha = 0.95
    max_iters = n * 10
    no_improve = 0
    while T > 1e-3 and no_improve < n*2:
        improved = False
        for _ in range(max_iters):
            i = rng.integers(0, n-2)
            j = rng.integers(i+2, n)
            delta = two_opt_delta(tour, i, j)
            if delta < 0 or rng.random() < np.exp(-delta/T):
                tour = np.concatenate([tour[:i+1], tour[i+1:j+1][::-1], tour[j+1:]])
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                    improved = True
                    no_improve = 0
        if not improved:
            no_improve += 1
        T *= alpha
        # Restart to best tour if stuck
        if no_improve >= n:
            tour = best_tour.copy()
            current_cost = best_cost
            no_improve = 0
    return best_tour