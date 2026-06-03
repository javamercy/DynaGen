import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    
    def tour_dist(tour):
        total = 0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i+1)%n]]
        return total
    
    def regret_insertion(seed=None):
        rng = np.random.RandomState(seed)
        tour = [0]
        first = np.argmin(distance_matrix[0][1:]) + 1
        tour.append(first)
        unvisited = set(range(n)) - set(tour)
        while unvisited:
            best_cost = {}
            sec_cost = {}
            best_pos = {}
            for city in unvisited:
                best = float('inf')
                sec = float('inf')
                best_idx = None
                for pos in range(len(tour)):
                    i = tour[pos]
                    j = tour[(pos+1)%len(tour)]
                    cost = distance_matrix[i][city] + distance_matrix[city][j] - distance_matrix[i][j]
                    if cost < best:
                        sec = best
                        best = cost
                        best_idx = pos
                    elif cost < sec:
                        sec = cost
                best_cost[city] = best
                sec_cost[city] = sec if sec != float('inf') else best
                best_pos[city] = best_idx
            regret = {c: sec_cost[c] - best_cost[c] for c in unvisited}
            candidates = list(unvisited)
            rng.shuffle(candidates)
            chosen = max(candidates, key=lambda c: (regret[c], -best_cost[c]))
            idx = best_pos[chosen]
            tour.insert(idx+1, chosen)
            unvisited.remove(chosen)
        return np.array(tour)
    
    best_tour = regret_insertion(seed=0)
    best_dist = tour_dist(best_tour)
    report_best_tour(best_tour)
    
    num_restarts = 10
    for restart in range(num_restarts):
        seed = restart * 12345 + 1
        tour = regret_insertion(seed)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    a = tour[i]
                    b = tour[(i+1)%n]
                    c = tour[j]
                    d = tour[(j+1)%n]
                    old = distance_matrix[a][b] + distance_matrix[c][d]
                    new = distance_matrix[a][c] + distance_matrix[b][d]
                    if new < old - 1e-10:
                        new_tour = np.concatenate([tour[:i+1], tour[i+1:j+1][::-1], tour[j+1:]])
                        new_dist = tour_dist(new_tour)
                        if new_dist < tour_dist(tour) - 1e-10:
                            tour = new_tour
                            improved = True
                            break
                if improved:
                    break
        dist = tour_dist(tour)
        if dist < best_dist - 1e-10:
            best_tour = tour
            best_dist = dist
            report_best_tour(best_tour)
    return best_tour