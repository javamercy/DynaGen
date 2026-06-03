import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    # Randomized nearest neighbor initial tour
    start = np.random.randint(n)
    tour = [start]
    visited = {start}
    while len(tour) < n:
        i = tour[-1]
        unvisited = [c for c in range(n) if c not in visited]
        # Probabilistic selection based on inverse distance
        inv_dist = 1.0 / (distance_matrix[i, unvisited] + 1e-10)
        prob = inv_dist / inv_dist.sum()
        next_city = np.random.choice(unvisited, p=prob)
        tour.append(next_city)
        visited.add(next_city)
    tour = np.array(tour)
    best_tour = tour.copy()
    best_dist = sum(distance_matrix[best_tour[k], best_tour[(k+1)%n]] for k in range(n))
    report_best_tour(best_tour)
    
    # Pheromone initialization
    mean_dist = np.mean(distance_matrix[distance_matrix > 0])
    tau0 = 1.0 / (n * mean_dist)
    pheromone = np.full((n, n), tau0)
    np.fill_diagonal(pheromone, 0)
    
    # Parameters
    n_ants = max(10, n // 20)
    n_iterations = 100
    alpha = 1.0
    beta = 5.0
    rho = 0.95
    Q = 1.0
    eta = 1.0 / (distance_matrix + np.eye(n))
    np.fill_diagonal(eta, 0)
    
    # Stepwise stagnation thresholds
    step_points = [0, n_iterations // 3, 2 * n_iterations // 3, n_iterations]
    step_values = [15, 8, 4]
    no_improve = 0
    
    for iteration in range(n_iterations):
        # Determine current max_no_improve
        max_no_improve = step_values[-1]
        for idx in range(len(step_points)-1):
            if step_points[idx] <= iteration < step_points[idx+1]:
                max_no_improve = step_values[idx]
                break
        
        tours = []
        dists = []
        for _ in range(n_ants):
            # Probabilistic construction
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
            tour_arr = np.array(tour)
            # 2-opt local search
            improved = True
            while improved:
                improved = False
                for i in range(n-1):
                    for j in range(i+1, n):
                        if j - i == 1:
                            continue
                        a, b = tour_arr[i], tour_arr[(i+1)%n]
                        c, d = tour_arr[j], tour_arr[(j+1)%n]
                        if distance_matrix[a, c] + distance_matrix[b, d] < distance_matrix[a, b] + distance_matrix[c, d]:
                            tour_arr[i+1:j+1] = tour_arr[i+1:j+1][::-1]
                            improved = True
            dist = sum(distance_matrix[tour_arr[k], tour_arr[(k+1)%n]] for k in range(n))
            tours.append(tour_arr)
            dists.append(dist)
            if dist < best_dist:
                best_dist = dist
                best_tour = tour_arr.copy()
                report_best_tour(best_tour)
                no_improve = 0
        
        # Evaporation
        pheromone *= (1 - rho)
        # Deposit
        for tour_arr, dist in zip(tours, dists):
            for k in range(n):
                i, j = tour_arr[k], tour_arr[(k+1)%n]
                pheromone[i, j] += Q / dist
                pheromone[j, i] += Q / dist
        # Elitist
        for k in range(n):
            i, j = best_tour[k], best_tour[(k+1)%n]
            pheromone[i, j] += Q / best_dist
            pheromone[j, i] += Q / best_dist
        
        no_improve += 1
        if no_improve >= max_no_improve:
            pheromone = np.full((n, n), tau0)
            np.fill_diagonal(pheromone, 0)
            no_improve = 0
    return best_tour