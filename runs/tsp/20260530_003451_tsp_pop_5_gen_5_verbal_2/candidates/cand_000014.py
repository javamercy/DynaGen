import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    
    def tour_cost(tour):
        cost = 0.0
        for k in range(n):
            cost += distance_matrix[tour[k], tour[(k+1)%n]]
        return cost
    
    def improve_2opt(tour):
        improved = True
        while improved:
            improved = False
            best_delta = 0.0
            best_i = best_j = -1
            for i in range(n):
                for j in range(i+2, n):
                    a = tour[i]
                    b = tour[(i+1)%n]
                    c = tour[j]
                    d = tour[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - (distance_matrix[a,b] + distance_matrix[c,d])
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            if best_delta < 0:
                i, j = best_i, best_j
                tour[i+1:j+1] = tour[i+1:j+1][::-1]
                improved = True
        return tour
    
    # Nearest neighbor construction
    tour = [0]
    unvisited = set(range(1, n))
    curr = 0
    while unvisited:
        nxt = min(unvisited, key=lambda c: distance_matrix[curr, c])
        tour.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    best_tour = np.array(tour, dtype=np.int32)
    best_cost = tour_cost(best_tour)
    report_best_tour(best_tour.copy())
    
    # Restart loop
    for _ in range(10):
        # Apply 2-opt improvement
        cur_tour = improve_2opt(best_tour.copy())
        cost = tour_cost(cur_tour)
        if cost < best_cost:
            best_cost = cost
            best_tour = cur_tour.copy()
            report_best_tour(best_tour.copy())
        
        # Double-bridge perturbation
        indices = np.random.choice(range(1, n-1), size=3, replace=False)
        i1, i2, i3 = sorted(indices)
        # Segments: [0..i1], [i1+1..i2], [i2+1..i3], [i3+1..n-1]
        # Rearrange to: seg0, seg2, seg1, seg3
        seg0 = best_tour[:i1+1]
        seg1 = best_tour[i1+1:i2+1]
        seg2 = best_tour[i2+1:i3+1]
        seg3 = best_tour[i3+1:]
        perturbed = np.concatenate([seg0, seg2, seg1, seg3])
        perturbed = improve_2opt(perturbed)
        cost = tour_cost(perturbed)
        if cost < best_cost:
            best_cost = cost
            best_tour = perturbed.copy()
            report_best_tour(best_tour.copy())
    
    return best_tour