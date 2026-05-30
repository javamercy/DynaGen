import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
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
    
    def tour_cost(t):
        cost = 0.0
        for k in range(n):
            cost += distance_matrix[t[k], t[(k+1)%n]]
        return cost
    
    best_cost = tour_cost(tour)
    report_best_tour(tour.copy())
    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_i = best_j = -1
        for i in range(n):
            for j in range(i+2, n):
                a, b = tour[i], tour[(i+1)%n]
                c, d = tour[j], tour[(j+1)%n]
                delta = distance_matrix[a,c] + distance_matrix[b,d] - (distance_matrix[a,b] + distance_matrix[c,d])
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        if best_delta < 0:
            i, j = best_i, best_j
            tour[i+1:j+1] = tour[i+1:j+1][::-1]
            cost = tour_cost(tour)
            if cost < best_cost:
                best_cost = cost
                report_best_tour(tour.copy())
            improved = True
    return tour