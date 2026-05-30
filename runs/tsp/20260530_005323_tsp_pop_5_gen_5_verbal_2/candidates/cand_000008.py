import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 1:
        return np.array([0])
    if n == 2:
        return np.array([0, 1])
    # Initial tour: two farthest cities
    max_dist = -1
    best_pair = (0, 1)
    for i in range(n):
        for j in range(i+1, n):
            d = distance_matrix[i][j]
            if d > max_dist:
                max_dist = d
                best_pair = (i, j)
    tour = list(best_pair)
    unvisited = set(range(n)) - set(tour)
    while unvisited:
        best_regret = -1
        best_city = None
        best_pos = None
        for k in unvisited:
            costs = []
            L = len(tour)
            for pos in range(L):
                i = tour[pos]
                j = tour[(pos+1)%L]
                cost = distance_matrix[i][k] + distance_matrix[k][j] - distance_matrix[i][j]
                costs.append(cost)
            sorted_costs = sorted(costs)
            best = sorted_costs[0]
            second_best = sorted_costs[1] if len(sorted_costs) > 1 else float('inf')
            regret = second_best - best
            if regret > best_regret:
                best_regret = regret
                best_city = k
                best_pos = costs.index(best)
        tour.insert(best_pos+1, best_city)
        unvisited.remove(best_city)
    tour_arr = np.array(tour)
    # Helper for total distance
    def tour_dist(t):
        d = 0.0
        for idx in range(n):
            d += distance_matrix[t[idx]][t[(idx+1)%n]]
        return d
    best_dist = tour_dist(tour_arr)
    report_best_tour(tour_arr)
    # Node insertion local search
    improved = True
    while improved:
        improved = False
        for i in range(n):
            city = tour[i]
            # Remove city i
            tour_no_i = tour[:i] + tour[i+1:]
            L = len(tour_no_i)
            best_pos = 0
            best_delta = float('inf')
            # Compute current contribution of edges involving city i
            prev_i = tour[(i-1)%n]
            next_i = tour[(i+1)%n]
            old_contrib = distance_matrix[prev_i][city] + distance_matrix[city][next_i]
            # For each insertion position (0..L) in tour_no_i
            for pos in range(L+1):
                if pos == 0:
                    a = tour_no_i[-1]
                    b = tour_no_i[0]
                    new_contrib = distance_matrix[a][city] + distance_matrix[city][b]
                elif pos == L:
                    a = tour_no_i[-1]
                    b = tour_no_i[0]
                    new_contrib = distance_matrix[a][city] + distance_matrix[city][b]
                else:
                    a = tour_no_i[pos-1]
                    b = tour_no_i[pos]
                    new_contrib = distance_matrix[a][city] + distance_matrix[city][b]
                delta = new_contrib - old_contrib
                if delta < best_delta:
                    best_delta = delta
                    best_pos = pos
            if best_delta < -1e-9:  # improvement
                # Insert city at best_pos
                tour = tour_no_i[:best_pos] + [city] + tour_no_i[best_pos:]
                improved = True
                new_tour_arr = np.array(tour)
                new_dist = tour_dist(new_tour_arr)
                if new_dist < best_dist:
                    best_dist = new_dist
                    report_best_tour(new_tour_arr)
                break  # restart loop after each change to avoid index confusion
        # End for
    return np.array(tour)