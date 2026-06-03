import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def nn_tour():
        tour = [0]
        visited = {0}
        cur = 0
        for _ in range(n - 1):
            best = min((v for v in range(n) if v not in visited), key=lambda v: distance_matrix[cur, v])
            tour.append(best)
            visited.add(best)
            cur = best
        return np.array(tour)

    tour = nn_tour()
    cost = lambda t: sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))
    current_cost = cost(tour)
    best_tour = tour.copy()
    best_cost = current_cost
    report_best_tour(tour)

    rng = np.random.default_rng()
    T = current_cost * 0.02
    if T == 0:
        T = 1.0
    alpha = 0.999
    epsilon = 1e-8
    reheat_threshold = n * 5
    no_improve = 0
    max_inner = n * 200

    while T > epsilon:
        for _ in range(max_inner):
            i = rng.integers(0, n - 2)
            j = rng.integers(i + 2, n)
            a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
            delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
            if delta < 0 or rng.random() < np.exp(-delta / T):
                tour = np.concatenate([tour[:i+1], tour[i+1:j+1][::-1], tour[j+1:]])
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            if no_improve >= reheat_threshold:
                T = T * 1.5
                no_improve = 0
        T *= alpha
        if T < epsilon:
            break

    return best_tour