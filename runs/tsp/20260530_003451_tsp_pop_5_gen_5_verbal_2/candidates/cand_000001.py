import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        return np.array([0])
    # Greedy initial tour
    tour = [0]
    unvisited = set(range(1, n))
    first_next = min(unvisited, key=lambda x: distance_matrix[0, x])
    tour.append(first_next)
    unvisited.remove(first_next)
    # Regret insertion for remaining
    while unvisited:
        best_costs = {}
        best_positions = {}
        second_best_costs = {}
        for city in unvisited:
            min_cost = np.inf
            second_min = np.inf
            best_pos = -1
            for i in range(len(tour)):
                prev = tour[i]
                next_node = tour[(i + 1) % len(tour)]
                cost = distance_matrix[prev, city] + distance_matrix[city, next_node] - distance_matrix[prev, next_node]
                if cost < min_cost:
                    second_min = min_cost
                    min_cost = cost
                    best_pos = i
                elif cost < second_min:
                    second_min = cost
            best_costs[city] = min_cost
            best_positions[city] = best_pos
            second_best_costs[city] = second_min
        # Choose city with max regret
        max_regret = -np.inf
        chosen_city = None
        for city in unvisited:
            regret = second_best_costs[city] - best_costs[city]
            if regret > max_regret:
                max_regret = regret
                chosen_city = city
        pos = best_positions[chosen_city]
        tour.insert(pos + 1, chosen_city)
        unvisited.remove(chosen_city)
    tour = np.array(tour, dtype=int)
    # Report initial tour
    best_tour = tour.copy()
    best_dist = total_distance(tour, distance_matrix)
    report_best_tour(tour)
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        n_nodes = len(tour)
        for i in range(n_nodes):
            for j in range(i + 2, n_nodes):
                new_tour = tour.copy()
                new_tour[i+1:j+1] = tour[i+1:j+1][::-1]
                new_dist = total_distance(new_tour, distance_matrix)
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_tour = new_tour.copy()
                    improved = True
                    report_best_tour(best_tour)
        tour = best_tour
    return best_tour

def total_distance(tour, dist):
    n = len(tour)
    d = 0
    for i in range(n):
        d += dist[tour[i], tour[(i + 1) % n]]
    return d