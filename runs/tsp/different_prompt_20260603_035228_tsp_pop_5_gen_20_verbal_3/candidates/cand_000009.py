import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    # Nearest neighbor initialization
    tour = np.zeros(n, dtype=int)
    unvisited = set(range(1, n))
    current = 0
    tour[0] = current
    for i in range(1, n):
        next_city = min(unvisited, key=lambda x: distance_matrix[current, x])
        tour[i] = next_city
        unvisited.remove(next_city)
        current = next_city
    best_tour = tour.copy()
    best_dist = total_distance(tour, distance_matrix)
    report_best_tour(best_tour)
    
    current_tour = tour.copy()
    current_dist = best_dist
    
    temperature = 10.0
    cooling_rate = 0.9995
    min_temperature = 1e-10
    
    while temperature > min_temperature:
        # 2-opt move: reverse a segment
        i, j = np.random.randint(1, n, 2)
        if i > j:
            i, j = j, i
        if i == j:
            j = (j + 1) % n
            if i > j:
                i, j = j, i
        new_tour = current_tour.copy()
        new_tour[i:j+1] = current_tour[i:j+1][::-1]
        new_dist = current_dist
        # Remove old edges, add new edges
        prev_i = current_tour[i-1] if i > 0 else current_tour[-1]
        curr_i = current_tour[i]
        curr_j = current_tour[j]
        next_j = current_tour[(j+1) % n]
        new_dist -= (distance_matrix[prev_i, curr_i] + distance_matrix[curr_j, next_j])
        new_dist += (distance_matrix[prev_i, curr_j] + distance_matrix[curr_i, next_j])
        
        if new_dist < current_dist or np.random.rand() < np.exp((current_dist - new_dist) / temperature):
            current_tour = new_tour
            current_dist = new_dist
            if current_dist < best_dist:
                best_dist = current_dist
                best_tour = current_tour.copy()
                report_best_tour(best_tour)
        temperature *= cooling_rate
    return best_tour

def total_distance(tour, dist):
    return dist[tour, np.roll(tour, -1)].sum()