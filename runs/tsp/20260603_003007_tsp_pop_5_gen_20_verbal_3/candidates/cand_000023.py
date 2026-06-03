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
    n_ants = 10
    n_iterations = 100
    alpha = 1.0
    beta = 2.0
    rho = 0.5
    Q = 1.0
    # heuristic
    with np.errstate(divide='ignore'):
        eta = 1.0 / (distance_matrix + np.eye(n))
    np.fill_diagonal(eta, 0)
    # stagnation detection
    stagnation = 0
    prev_best = best_dist
    for it in range(n_iterations):
        for _ in range(n_ants):
            # construct tour
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
            # evaluate
            dist = sum(distance_matrix[tour_arr[k], tour_arr[(k+1)%n]] for k in range(n))
            if dist < best_dist:
                best_dist = dist
                best_tour = tour_arr.copy()
                report_best_tour(best_tour)
                stagnation = 0
            # pheromone deposit
            for k in range(n):
                i, j = tour_arr[k], tour_arr[(k+1)%n]
                pheromone[i, j] += Q / dist
                pheromone[j, i] += Q / dist
        # evaporation
        pheromone *= (1 - rho)
        # adaptive restart
        if best_dist == prev_best:
            stagnation += 1
        else:
            stagnation = 0
            prev_best = best_dist
        if stagnation >= 20:
            pheromone = np.full((n, n), tau0)
            np.fill_diagonal(pheromone, 0)
            stagnation = 0
    return best_tour