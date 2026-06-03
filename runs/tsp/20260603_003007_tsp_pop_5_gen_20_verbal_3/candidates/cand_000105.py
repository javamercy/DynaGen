import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    # Nearest neighbor initial tour
    start = 0
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    cur = start
    while unvisited:
        next_city = min(unvisited, key=lambda x: distance_matrix[cur, x])
        tour.append(next_city)
        unvisited.remove(next_city)
        cur = next_city
    best_tour = np.array(tour)
    best_dist = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour)
    # Parameters
    n_ants = 20
    n_iterations = 200
    alpha = 1.0
    beta = 5.0
    rho = 0.95
    Q = 1.0
    MAX_STAGNATION = 10
    # Pheromone and heuristic initialization
    mean_dist = np.mean(distance_matrix[distance_matrix > 0])
    tau0 = 1.0 / (n * mean_dist)
    pheromone = np.full((n, n), tau0)
    np.fill_diagonal(pheromone, 0)
    eta = 1.0 / (distance_matrix + np.eye(n))
    np.fill_diagonal(eta, 0)
    stagnation_counter = 0
    # Main loop
    for iteration in range(n_iterations):
        tours = []
        dists = []
        # Build ant tours
        for _ in range(n_ants):
            tour = [np.random.randint(n)]
            visited = set(tour)
            while len(tour) < n:
                i = tour[-1]
                tau = pheromone[i, :]
                prob = (tau ** alpha) * (eta[i, :] ** beta)
                prob[list(visited)] = 0
                prob_sum = prob.sum()
                if prob_sum == 0:
                    choices = [c for c in range(n) if c not in visited]
                    next_city = np.random.choice(choices)
                else:
                    prob = prob / prob_sum
                    next_city = np.random.choice(n, p=prob)
                tour.append(next_city)
                visited.add(next_city)
            # 2-opt local search
            improved = True
            tour_arr = np.array(tour)
            while improved:
                improved = False
                for i in range(n-1):
                    for j in range(i+2, n):
                        if distance_matrix[tour_arr[i], tour_arr[j]] + distance_matrix[tour_arr[(i+1)%n], tour_arr[(j+1)%n]] < distance_matrix[tour_arr[i], tour_arr[(i+1)%n]] + distance_matrix[tour_arr[j], tour_arr[(j+1)%n]]:
                            tour_arr[i+1:j+1] = tour_arr[i+1:j+1][::-1]
                            improved = True
            dist = sum(distance_matrix[tour_arr[k], tour_arr[(k+1)%n]] for k in range(n))
            tours.append(tour_arr)
            dists.append(dist)
            if dist < best_dist:
                best_dist = dist
                best_tour = tour_arr.copy()
                report_best_tour(best_tour)
                stagnation_counter = 0
        # Pheromone evaporation and deposit
        pheromone *= (1 - rho)
        for t_arr, d in zip(tours, dists):
            for k in range(n):
                i, j = t_arr[k], t_arr[(k+1)%n]
                pheromone[i, j] += Q / d
                pheromone[j, i] += Q / d
        # Elitist reinforcement
        for k in range(n):
            i, j = best_tour[k], best_tour[(k+1)%n]
            pheromone[i, j] += Q / best_dist
            pheromone[j, i] += Q / best_dist
        # Stagnation check and ILS
        stagnation_counter += 1
        if stagnation_counter >= MAX_STAGNATION:
            pheromone = np.full((n, n), tau0)
            np.fill_diagonal(pheromone, 0)
            for _ in range(3):
                new_tour = best_tour.copy()
                i, j = np.random.choice(n, 2, replace=False)
                new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
                # 2-opt on new_tour
                improved = True
                while improved:
                    improved = False
                    for a in range(n-1):
                        for b in range(a+2, n):
                            if distance_matrix[new_tour[a], new_tour[b]] + distance_matrix[new_tour[(a+1)%n], new_tour[(b+1)%n]] < distance_matrix[new_tour[a], new_tour[(a+1)%n]] + distance_matrix[new_tour[b], new_tour[(b+1)%n]]:
                                new_tour[a+1:b+1] = new_tour[a+1:b+1][::-1]
                                improved = True
                new_dist = sum(distance_matrix[new_tour[k], new_tour[(k+1)%n]] for k in range(n))
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_tour = new_tour.copy()
                    report_best_tour(best_tour)
                    break
            stagnation_counter = 0
    return best_tour