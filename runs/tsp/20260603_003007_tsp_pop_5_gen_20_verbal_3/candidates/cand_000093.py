import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    # nearest neighbor initial tour
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
    # pheromone initialization
    mean_dist = np.mean(distance_matrix[distance_matrix > 0])
    tau0 = 1.0 / (n * mean_dist)
    pheromone = np.full((n, n), tau0)
    np.fill_diagonal(pheromone, 0)
    # parameters
    n_ants = max(15, n // 8)
    n_iterations = 200
    alpha = 1.0
    beta = 5.0
    rho = 0.95
    Q = 1.0
    # heuristic
    eta = 1.0 / (distance_matrix + np.eye(n))
    np.fill_diagonal(eta, 0)
    # stepwise stagnation threshold: earlier stages more sensitive
    step_points = [0, n_iterations // 3, 2 * n_iterations // 3, n_iterations]
    step_values = [15, 10, 5]
    no_improve = 0
    # main loop
    for iteration in range(n_iterations):
        # determine max_no_improve for current iteration
        for idx in range(len(step_points)-1):
            if step_points[idx] <= iteration < step_points[idx+1]:
                max_no_improve = step_values[idx]
                break
        tours = []
        dists = []
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
        # evaporation
        pheromone *= (1 - rho)
        # deposit pheromone from ants
        for tour_arr, dist in zip(tours, dists):
            for k in range(n):
                i, j = tour_arr[k], tour_arr[(k+1)%n]
                pheromone[i, j] += Q / dist
                pheromone[j, i] += Q / dist
        # elitist reinforcement on best tour
        for k in range(n):
            i, j = best_tour[k], best_tour[(k+1)%n]
            pheromone[i, j] += Q / best_dist
            pheromone[j, i] += Q / best_dist
        # stagnation check with iterated local search
        no_improve += 1
        if no_improve >= max_no_improve:
            pheromone = np.full((n, n), tau0)
            np.fill_diagonal(pheromone, 0)
            # adaptive ILS: up to 5 perturbations
            improved_ils = False
            for _ in range(5):
                new_tour = best_tour.copy()
                i, j = np.random.choice(n, 2, replace=False)
                new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
                improved = True
                while improved:
                    improved = False
                    for a in range(n-1):
                        for b in range(a+2, n):
                            if distance_matrix[new_tour[a], new_tour[b]] + distance_matrix[new_tour[a+1], new_tour[(b+1)%n]] < distance_matrix[new_tour[a], new_tour[a+1]] + distance_matrix[new_tour[b], new_tour[(b+1)%n]]:
                                new_tour[a+1:b+1] = new_tour[a+1:b+1][::-1]
                                improved = True
                new_dist = sum(distance_matrix[new_tour[k], new_tour[(k+1)%n]] for k in range(n))
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_tour = new_tour.copy()
                    report_best_tour(best_tour)
                    improved_ils = True
                    break
            if not improved_ils:
                # random restart: nearest neighbor from random start
                start_rnd = np.random.randint(n)
                unvisited = set(range(n))
                unvisited.remove(start_rnd)
                tour = [start_rnd]
                cur = start_rnd
                while unvisited:
                    next_city = min(unvisited, key=lambda x: distance_matrix[cur, x])
                    tour.append(next_city)
                    unvisited.remove(next_city)
                    cur = next_city
                new_tour = np.array(tour)
                new_dist = sum(distance_matrix[new_tour[k], new_tour[(k+1)%n]] for k in range(n))
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_tour = new_tour.copy()
                    report_best_tour(best_tour)
            no_improve = 0
    return best_tour