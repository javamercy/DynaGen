import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    # Nearest neighbor construction
    tour = [0]
    unvisited = set(range(1, n))
    curr = 0
    while unvisited:
        nxt = min(unvisited, key=lambda c: distance_matrix[curr, c])
        tour.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    tour = np.array(tour, dtype=np.int32)
    best_tour = tour.copy()
    best_cost = np.inf
    
    def tour_cost(t):
        cost = 0.0
        for k in range(n):
            cost += distance_matrix[t[k], t[(k+1)%n]]
        return cost
    
    def improve(t):
        improved = True
        while improved:
            improved = False
            best_delta = 0.0
            best_i = best_j = -1
            for i in range(n):
                for j in range(i+2, n):
                    a, b = t[i], t[(i+1)%n]
                    c, d = t[j], t[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - (distance_matrix[a,b] + distance_matrix[c,d])
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            if best_delta < 0:
                i, j = best_i, best_j
                if i+1 < j+1:
                    t[i+1:j+1] = t[i+1:j+1][::-1]
                else:
                    # wrap-around case, reverse from i+1 to end then from start to j
                    seg = np.concatenate([t[i+1:], t[:j+1]])
                    seg = seg[::-1]
                    t[i+1:] = seg[:n-(i+1)]
                    t[:j+1] = seg[n-(i+1):]
                improved = True
        return t
    
    def double_bridge(t):
        # random split into 4 segments
        a = np.random.randint(1, n//4)
        b = a + np.random.randint(1, n//4)
        c = b + np.random.randint(1, n//4)
        # ensure a,b,c are valid indices
        if c >= n:
            c = n - 1
        if b >= c:
            b = c - 1
        if a >= b:
            a = b - 1
        # reassemble: [0..a-1] + [b..c-1] + [a..b-1] + [c..n-1]
        new_tour = np.concatenate([t[:a], t[b:c], t[a:b], t[c:]])
        return new_tour
    
    # Initial best
    cost = tour_cost(tour)
    if cost < best_cost:
        best_cost = cost
        best_tour = tour.copy()
        report_best_tour(best_tour)
    
    # Outer loop with perturbation
    for restart in range(10):  # limit perturbations to avoid timeout
        tour = improve(tour)
        cost = tour_cost(tour)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour.copy()
            report_best_tour(best_tour)
        # apply double bridge if not first iteration
        if restart < 9:
            tour = double_bridge(tour)
    return best_tour