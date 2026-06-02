def solve_tsp(distance_matrix):
    import numpy as np
    import random
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=int)
        report_best_tour(tour)
        return tour
    # farthest insertion construction
    unvisited = set(range(n))
    tour = [random.randint(0, n-1)]
    unvisited.remove(tour[0])
    farthest = max(unvisited, key=lambda c: distance_matrix[tour[0], c])
    tour.append(farthest)
    unvisited.remove(farthest)
    while unvisited:
        best_city = None
        best_pos = None
        best_dist = -1
        for city in unvisited:
            max_inc = -1
            best_pos_for_city = None
            for i in range(len(tour)):
                j = (i+1) % len(tour)
                inc = distance_matrix[tour[i], city] + distance_matrix[city, tour[j]] - distance_matrix[tour[i], tour[j]]
                if inc > max_inc:
                    max_inc = inc
                    best_pos_for_city = i+1
            if max_inc > best_dist:
                best_dist = max_inc
                best_city = city
                best_pos = best_pos_for_city
        tour.insert(best_pos, best_city)
        unvisited.remove(best_city)
    tour = np.array(tour, dtype=int)
    def compute_dist(t):
        d = 0.0
        for i in range(n):
            d += distance_matrix[t[i], t[(i+1)%n]]
        return d
    best_tour = tour.copy()
    best_dist = compute_dist(best_tour)
    report_best_tour(best_tour)
    def two_opt(t, d):
        improved = True
        while improved:
            improved = False
            for i in range(n-1):
                for j in range(i+2, n):
                    a = t[i]
                    b = t[(i+1)%n]
                    c = t[j]
                    d_node = t[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d_node] - distance_matrix[a,b] - distance_matrix[c,d_node]
                    if delta < -1e-10:
                        new_t = t.copy()
                        new_t[i+1:j+1] = t[j:i:-1]
                        t = new_t
                        d += delta
                        improved = True
                        break
                if improved:
                    break
        return t, d
    best_tour, best_dist = two_opt(best_tour, best_dist)
    report_best_tour(best_tour)
    # ILS with double-bridge and 2-opt
    max_iter = 5
    for _ in range(max_iter):
        t = best_tour.copy()
        indices = sorted(random.sample(range(n), 4))
        a,b,c,d = indices
        new_t = np.concatenate([t[:a], t[c:d], t[b:c], t[a:b], t[d:]]).astype(int)
        new_d = compute_dist(new_t)
        new_t, new_d = two_opt(new_t, new_d)
        if new_d < best_dist - 1e-10:
            best_dist = new_d
            best_tour = new_t.copy()
            report_best_tour(best_tour)
    return best_tour