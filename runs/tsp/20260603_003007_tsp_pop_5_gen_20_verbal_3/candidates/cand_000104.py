import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    best_dist = np.inf
    best_tour = None
    num_restarts = max(5, min(10, int(np.sqrt(n))))
    for _ in range(num_restarts):
        start = np.random.randint(n)
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        cur = start
        while unvisited:
            next_city = min(unvisited, key=lambda x: distance_matrix[cur, x])
            tour.append(next_city)
            unvisited.remove(next_city)
            cur = next_city
        tour = np.array(tour, dtype=int)
        curr_dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
        
        def two_opt(t, dist):
            improved = True
            while improved:
                improved = False
                for i in range(n-2):
                    for j in range(i+2, n):
                        if j - i == 1:
                            continue
                        a, b = t[i], t[(i+1)%n]
                        c, d = t[j], t[(j+1)%n]
                        delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                        if delta < -1e-12:
                            t[i+1:j+1] = t[i+1:j+1][::-1]
                            improved = True
                            dist += delta
            return t, dist
        
        tour, curr_dist = two_opt(tour, curr_dist)
        if curr_dist < best_dist:
            best_dist = curr_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        
        num_perturbations = n  # adaptive: linear with n
        for _ in range(num_perturbations):
            i, j = np.random.choice(n, 2, replace=False)
            new_tour = tour.copy()
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            new_dist = sum(distance_matrix[new_tour[i], new_tour[(i+1)%n]] for i in range(n))
            new_tour, new_dist = two_opt(new_tour, new_dist)
            if new_dist < best_dist:
                best_dist = new_dist
                best_tour = new_tour.copy()
                report_best_tour(best_tour)
                tour = new_tour
                curr_dist = new_dist
            elif new_dist < curr_dist:
                tour = new_tour
                curr_dist = new_dist
    return best_tour