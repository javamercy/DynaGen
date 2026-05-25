import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = len(distance_matrix)
    if n <= 3:
        tour = list(range(n))
        np.random.shuffle(tour)
        return np.array(tour)
    
    # Nearest insertion construction (O(n^2))
    start = np.random.randint(n)
    tour = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    last = start
    while unvisited:
        # find nearest unvisited to last
        best_city = None
        best_dist = np.inf
        for city in unvisited:
            d = distance_matrix[last, city]
            if d < best_dist:
                best_dist = d
                best_city = city
        tour.append(best_city)
        unvisited.remove(best_city)
        last = best_city
    tour_arr = np.array(tour)
    best_dist = total_distance(tour_arr, distance_matrix)
    report_best_tour(tour_arr)
    
    # 2-opt with delta calculations, limited by budget (number of move attempts)
    checks = 0
    improved = True
    while improved and checks < budget:
        improved = False
        for i in range(n-1):
            for j in range(i+2, n):
                if checks >= budget:
                    break
                checks += 1
                # compute delta
                a = i
                b = (i+1) % n
                c = j
                d = (j+1) % n
                delta = distance_matrix[tour_arr[a], tour_arr[c]] + distance_matrix[tour_arr[b], tour_arr[d]] - distance_matrix[tour_arr[a], tour_arr[b]] - distance_matrix[tour_arr[c], tour_arr[d]]
                if delta < -1e-12:
                    # reverse segment
                    tour_arr = np.concatenate([tour_arr[:i+1], tour_arr[i+1:j+1][::-1], tour_arr[j+1:]])
                    best_dist += delta
                    report_best_tour(tour_arr)
                    improved = True
                    break
            if improved or checks >= budget:
                break
    return tour_arr

def total_distance(tour, dist):
    n = len(tour)
    return sum(dist[tour[i], tour[(i+1)%n]] for i in range(n))