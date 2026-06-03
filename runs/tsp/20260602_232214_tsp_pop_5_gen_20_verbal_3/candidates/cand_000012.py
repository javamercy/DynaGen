import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # start with farthest pair
    i, j = np.unravel_index(np.argmax(distance_matrix), (n, n))
    tour = [i, j]
    unvisited = set(range(n)) - {i, j}
    
    while unvisited:
        best_costs = {}
        second_costs = {}
        min_dists = {}
        best_positions = {}
        for city in unvisited:
            best = float('inf')
            second = float('inf')
            pos = 0
            min_dist = float('inf')
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
                # track minimum distance to tour nodes
                d = distance_matrix[city, a]
                if d < min_dist:
                    min_dist = d
                d = distance_matrix[city, b]
                if d < min_dist:
                    min_dist = d
            best_costs[city] = best
            second_costs[city] = second if second != float('inf') else best
            best_positions[city] = pos
            min_dists[city] = min_dist
        
        # choose city maximizing regret + min_dist, tie-break by lower best cost
        def score(c):
            regret = second_costs[c] - best_costs[c]
            return regret + min_dists[c]
        best_score = max(score(c) for c in unvisited)
        candidates = [c for c in unvisited if score(c) == best_score]
        chosen = min(candidates, key=lambda c: best_costs[c])
        
        pos = best_positions[chosen]
        tour.insert(pos, chosen)
        unvisited.remove(chosen)
    
    tour_arr = np.array(tour)
    report_best_tour(tour_arr)
    
    # 2-opt improvement
    improved = True
    best_tour = tour_arr.copy()
    best_dist = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n]] for i in range(n))
    while improved:
        improved = False
        for i in range(n-2):
            for j in range(i+2, n):
                a = best_tour[i]
                b = best_tour[i+1]
                c = best_tour[j]
                d = best_tour[(j+1)%n]
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