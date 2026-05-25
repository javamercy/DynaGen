import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = len(distance_matrix)
    if n <= 3:
        tour = list(range(n))
        np.random.shuffle(tour)
        return np.array(tour)
    
    # Start with two random cities
    start = np.random.randint(n)
    tour = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    # Add a second city randomly
    second = np.random.choice(list(unvisited))
    tour.append(second)
    unvisited.remove(second)
    
    # Regret insertion
    while unvisited:
        best_city = None
        best_pos = None
        best_regret = -np.inf
        for city in unvisited:
            costs = []
            for pos in range(len(tour)+1):
                if pos == 0:
                    cost = distance_matrix[city, tour[0]] + distance_matrix[tour[-1], city] - distance_matrix[tour[-1], tour[0]]
                elif pos == len(tour):
                    cost = distance_matrix[tour[-1], city] + distance_matrix[city, tour[0]] - distance_matrix[tour[-1], tour[0]]
                else:
                    cost = distance_matrix[tour[pos-1], city] + distance_matrix[city, tour[pos]] - distance_matrix[tour[pos-1], tour[pos]]
                costs.append(cost)
            sorted_costs = sorted(costs)
            best = sorted_costs[0]
            second_best = sorted_costs[1] if len(sorted_costs) > 1 else best
            regret = second_best - best
            if regret > best_regret:
                best_regret = regret
                best_city = city
                best_pos = np.argmin(costs)
        # Insert best_city at best_pos
        tour.insert(best_pos, best_city)
        unvisited.remove(best_city)
    
    tour_arr = np.array(tour)
    best_dist = total_distance(tour_arr, distance_matrix)
    report_best_tour(tour_arr)
    
    # 2-opt improvement
    improved = True
    iters = 0
    max_iters = max(1, budget // 2)
    while improved and iters < max_iters:
        improved = False
        for i in range(n-1):
            for j in range(i+2, n):
                if i == 0 and j == n-1:
                    continue
                new_tour = tour_arr.copy()
                new_tour[i:j+1] = np.flip(tour_arr[i:j+1])
                new_dist = total_distance(new_tour, distance_matrix)
                if new_dist < best_dist:
                    tour_arr = new_tour
                    best_dist = new_dist
                    report_best_tour(tour_arr)
                    improved = True
                    break
            if improved:
                break
        iters += 1
    return tour_arr

def total_distance(tour, dist):
    n = len(tour)
    return sum(dist[tour[i], tour[(i+1)%n]] for i in range(n))