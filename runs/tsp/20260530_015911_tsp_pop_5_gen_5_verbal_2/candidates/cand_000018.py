import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    # farthest insertion construction
    tour = [0]
    unvisited = set(range(1, n))
    farthest = max(unvisited, key=lambda x: distance_matrix[0, x])
    tour.append(farthest)
    unvisited.remove(farthest)
    while unvisited:
        best_dist = -1
        best_city = None
        for city in unvisited:
            min_dist = min(distance_matrix[city, t] for t in tour)
            if min_dist > best_dist:
                best_dist = min_dist
                best_city = city
        best_pos = None
        best_inc = float('inf')
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            inc = distance_matrix[tour[i], best_city] + distance_matrix[best_city, tour[j]] - distance_matrix[tour[i], tour[j]]
            if inc < best_inc:
                best_inc = inc
                best_pos = j
        tour.insert(best_pos, best_city)
        unvisited.remove(best_city)
    tour = np.array(tour, dtype=np.int32)
    best_tour = tour.copy()
    best_dist = distance_matrix[tour[-1], tour[0]] + np.sum(distance_matrix[tour[:-1], tour[1:]])
    report_best_tour(best_tour)

    def calc_dist(t):
        return distance_matrix[t[-1], t[0]] + np.sum(distance_matrix[t[:-1], t[1:]])

    def two_opt(t):
        improved = True
        while improved:
            improved = False
            for i in range(n-1):
                for j in range(i+2, n):
                    a, b = t[i], t[(i+1)%n]
                    c, d = t[j], t[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-12:
                        t[i+1:j+1] = np.flip(t[i+1:j+1])
                        improved = True
                        break
                if improved:
                    break
        return t

    tour = two_opt(tour)
    d = calc_dist(tour)
    if d < best_dist - 1e-12:
        best_dist = d
        best_tour = tour.copy()
        report_best_tour(best_tour)

    # simple ILS with random inversion perturbation
    for _ in range(5):
        i = np.random.randint(0, n-2)
        j = np.random.randint(i+2, n)
        pert = tour.copy()
        pert[i:j] = np.flip(pert[i:j])
        new_tour = two_opt(pert)
        new_d = calc_dist(new_tour)
        if new_d < d - 1e-12:
            tour = new_tour
            d = new_d
            if d < best_dist - 1e-12:
                best_dist = d
                best_tour = tour.copy()
                report_best_tour(best_tour)

    # final 2-opt polish on best
    final_tour = two_opt(best_tour.copy())
    final_d = calc_dist(final_tour)
    if final_d < best_dist - 1e-12:
        best_tour = final_tour
        report_best_tour(best_tour)

    return best_tour