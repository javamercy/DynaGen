import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # find farthest pair
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
        # choose city with max regret, tie-break by lower best cost
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
    # 2-opt improvement
    n_cities = n
    improved = True
    best_tour = tour_arr.copy()
    best_dist = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n_cities]] for i in range(n_cities))
    while improved:
        improved = False
        for i in range(n_cities-2):
            for j in range(i+2, n_cities):
                a = best_tour[i]
                b = best_tour[i+1]
                c = best_tour[j]
                d = best_tour[(j+1)%n_cities]
                delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                if delta < -1e-10:
                    new_tour = best_tour.copy()
                    new_tour[i+1:j+1] = best_tour[j:i:-1]
                    new_dist = best_dist + delta
                    best_tour = new_tour
                    best_dist = new_dist
                    improved = True
                    report_best_tour(best_tour)
    return best_tour