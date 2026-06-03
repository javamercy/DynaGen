import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # farthest pair
    i, j = np.unravel_index(np.argmax(distance_matrix), (n, n))
    tour = [i, j]
    unvisited = set(range(n)) - {i, j}
    # regret insertion
    while unvisited:
        best_cost = {}
        second_cost = {}
        best_pos = {}
        for city in unvisited:
            best = float('inf')
            second = float('inf')
            pos = 0
            for k in range(len(tour)):
                a = tour[k]
                b = tour[(k+1) % len(tour)]
                cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                if cost < best:
                    second = best
                    best = cost
                    pos = k+1
                elif cost < second:
                    second = cost
            best_cost[city] = best
            second_cost[city] = second if second != float('inf') else best
            best_pos[city] = pos
        max_regret = max(second_cost[c] - best_cost[c] for c in unvisited)
        candidates = [c for c in unvisited if second_cost[c] - best_cost[c] == max_regret]
        if len(candidates) > 1:
            chosen = min(candidates, key=lambda c: best_cost[c])
        else:
            chosen = candidates[0]
        pos = best_pos[chosen]
        tour.insert(pos, chosen)
        unvisited.remove(chosen)
    tour_arr = np.array(tour)
    report_best_tour(tour_arr)

    def two_opt(t):
        n_cities = len(t)
        improved = True
        best_t = t.copy()
        best_dist = sum(distance_matrix[best_t[i], best_t[(i+1)%n_cities]] for i in range(n_cities))
        while improved:
            improved = False
            for i in range(n_cities-2):
                for j in range(i+2, n_cities):
                    a = best_t[i]
                    b = best_t[i+1]
                    c = best_t[j]
                    d = best_t[(j+1)%n_cities]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-10:
                        best_t[i+1:j+1] = best_t[j:i:-1]
                        best_dist += delta
                        improved = True
                        report_best_tour(best_t)
        return best_t, best_dist

    best_tour, best_dist = two_opt(tour_arr)

    # Iterated local search with double-bridge perturbation
    if n > 3:
        for _ in range(10):
            # double-bridge
            i = np.random.randint(1, n-2)
            j = np.random.randint(i+1, n-1)
            k = np.random.randint(j+1, n)
            perturbed = np.concatenate([best_tour[:i], best_tour[j:k], best_tour[i:j], best_tour[k:]])
            new_tour, new_dist = two_opt(perturbed)
            if new_dist < best_dist - 1e-10:
                best_tour = new_tour
                best_dist = new_dist
    report_best_tour(best_tour)
    return best_tour