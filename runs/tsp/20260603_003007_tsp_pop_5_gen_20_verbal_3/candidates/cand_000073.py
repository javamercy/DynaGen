import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    # random initial tour as baseline
    best_tour = np.random.permutation(n)
    best_dist = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour)
    # pheromone initialization
    mean_dist = np.mean(distance_matrix[distance_matrix > 0])
    tau0 = 1.0 / (n * mean_dist)
    pheromone = np.full((n, n), tau0)
    np.fill_diagonal(pheromone, 0)
    # parameters
    n_ants = max(10, n // 10)
    n_iterations = 200
    alpha = 1.0
    beta = 5.0
    rho = 0.95
    Q = 1.0
    # heuristic
    eta = 1.0 / (distance_matrix + np.eye(n))
    np.fill_diagonal(eta, 0)
    # stepwise stagnation thresholds
    step_points = [0, n_iterations // 3, 2 * n_iterations // 3, n_iterations]
    step_values = [20, 10, 5]
    no_improve = 0
    # precompute nearest neighbor for each starting city
    def nearest_neighbor(start):
        tour = [start]
        visited = {start}
        for _ in range(n-1):
            i = tour[-1]
            # find nearest unvisited
            min_dist = np.inf
            best_j = -1
            for j in range(n):
                if j not in visited and distance_matrix[i,j] < min_dist:
                    min_dist = distance_matrix[i,j]
                    best_j = j
            tour.append(best_j)
            visited.add(best_j)
        return np.array(tour)
    # main loop
    for iteration in range(n_iterations):
        # determine max_no_improve
        for idx in range(len(step_points)-1):
            if step_points[idx] <= iteration < step_points[idx+1]:
                max_no_improve = step_values[idx]
                break
        tours = []
        dists = []
        for ant in range(n_ants):
            # choose construction method randomly
            method = np.random.choice(['random', 'nn', 'ant'], p=[0.2, 0.2, 0.6])
            if method == 'random':
                tour_arr = np.random.permutation(n)
            elif method == 'nn':
                start = np.random.randint(n)
                tour_arr = nearest_neighbor(start)
            else:  # ant system
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
        # pheromone update
        pheromone *= (1 - rho)
        for tour_arr, dist in zip(tours, dists):
            for k in range(n):
                i, j = tour_arr[k], tour_arr[(k+1)%n]
                pheromone[i, j] += Q / dist
                pheromone[j, i] += Q / dist
        # elitist reinforcement
        for k in range(n):
            i, j = best_tour[k], best_tour[(k+1)%n]
            pheromone[i, j] += Q / best_dist
            pheromone[j, i] += Q / best_dist
        # stagnation restart
        no_improve += 1
        if no_improve >= max_no_improve:
            pheromone = np.full((n, n), tau0)
            np.fill_diagonal(pheromone, 0)
            no_improve = 0
    return best_tour