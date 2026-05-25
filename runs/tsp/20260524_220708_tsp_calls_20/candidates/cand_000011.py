import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n)
        np.random.shuffle(tour)
        return tour

    best_tour = None
    best_dist = np.inf
    total_attempts = 0

    while total_attempts < budget:
        # Regret-2 construction
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            max_regret = -1
            chosen_city = None
            chosen_pos = None
            for city in unvisited:
                best_cost = np.inf
                second_best = np.inf
                best_pos = None
                for pos in range(len(tour) + 1):
                    if pos == 0:
                        prev = tour[-1]
                        nxt = tour[0]
                    elif pos == len(tour):
                        prev = tour[-1]
                        nxt = tour[0]
                    else:
                        prev = tour[pos-1]
                        nxt = tour[pos]
                    cost = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
                    if cost < best_cost:
                        second_best = best_cost
                        best_cost = cost
                        best_pos = pos
                    elif cost < second_best:
                        second_best = cost
                regret = second_best - best_cost
                if regret > max_regret:
                    max_regret = regret
                    chosen_city = city
                    chosen_pos = best_pos
            tour.insert(chosen_pos, chosen_city)
            unvisited.remove(chosen_city)

        tour_arr = np.array(tour)
        dist = total_distance(tour_arr, distance_matrix)
        if dist < best_dist:
            best_dist = dist
            best_tour = tour_arr.copy()
            report_best_tour(best_tour)

        # 2-opt with budget control
        improved = True
        while improved and total_attempts < budget:
            improved = False
            for i in range(n-1):
                for j in range(i+2, n):
                    if total_attempts >= budget:
                        break
                    total_attempts += 1
                    a = i
                    b = (i+1) % n
                    c = j
                    d = (j+1) % n
                    delta = (distance_matrix[tour_arr[a], tour_arr[c]] +
                             distance_matrix[tour_arr[b], tour_arr[d]] -
                             distance_matrix[tour_arr[a], tour_arr[b]] -
                             distance_matrix[tour_arr[c], tour_arr[d]])
                    if delta < -1e-12:
                        tour_arr = np.concatenate([tour_arr[:i+1],
                                                   tour_arr[i+1:j+1][::-1],
                                                   tour_arr[j+1:]])
                        dist += delta
                        if dist < best_dist:
                            best_dist = dist
                            best_tour = tour_arr.copy()
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved or total_attempts >= budget:
                    break

    return best_tour

def total_distance(tour, dist):
    n = len(tour)
    return sum(dist[tour[i], tour[(i+1)%n]] for i in range(n))