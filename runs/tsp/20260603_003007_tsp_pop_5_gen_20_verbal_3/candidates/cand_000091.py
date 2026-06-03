import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    # Regret-based insertion initial tour
    start = np.random.randint(n)
    tour = [start]
    unvisited = list(range(n))
    unvisited.remove(start)
    while unvisited:
        best_insertion = {}
        for city in unvisited:
            costs = []
            for pos in range(len(tour)):
                a = tour[pos]
                b = tour[(pos+1) % len(tour)]
                cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                costs.append(cost)
            costs.sort()
            best = costs[0]
            second_best = costs[1] if len(costs) > 1 else best
            regret = second_best - best
            best_insertion[city] = (regret, best, costs.index(best))  # using first occurrence index
        # Choose city with max regret
        city = max(unvisited, key=lambda x: best_insertion[x][0])
        _, _, pos = best_insertion[city]
        tour.insert(pos, city)
        unvisited.remove(city)
    best_tour = np.array(tour)
    best_dist = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour)
    # Pheromone initialization
    mean_dist = np.mean(distance_matrix[distance_matrix > 0])
    tau0 = 1.0 / (n * mean_dist)
    pheromone = np.full((n, n), tau0)
    np.fill_diagonal(pheromone, 0)
    # Parameters
    n_ants = max(10, n // 10)
    n_iterations = 200
    alpha = 1.0
    beta = 5.0
    rho = 0.95
    Q = 1.0
    eta = 1.0 / (distance_matrix + np.eye(n))
    np.fill_diagonal(eta, 0)
    step_points = [0, n_iterations // 3, 2 * n_iterations // 3, n_iterations]
    step_values = [20, 10, 5]
    no_improve = 0
    for iteration in range(n_iterations):
        for idx in range(len(step_points)-1):
            if step_points[idx] <= iteration < step_points[idx+1]:
                max_no_improve = step_values[idx]
                break
        tours = []
        dists = []
        for _ in range(n_ants):
            tour_arr = np.array([np.random.randint(n)])
            visited = set(tour_arr)
            while len(tour_arr) < n:
                i = tour_arr[-1]
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
                tour_arr = np.append(tour_arr, next_city)
                visited.add(next_city)
            # 2-opt local search
            improved = True
            while improved:
                improved = False
                for i in range(n-1):
                    for j in range(i+1, n):
                        if j - i == 1:
                            continue
                        a = tour_arr[i]
                        b = tour_arr[(i+1)%n]
                        c = tour_arr[j]
                        d = tour_arr[(j+1)%n]
                        if distance_matrix[a, c] + distance_matrix[b, d] < distance_matrix[a, b] + distance_matrix[c, d]:
                            tour_arr[i+1:j+1] = tour_arr[i+1:j+1][::-1]
                            improved = True
            dist = sum(distance_matrix[tour_arr[k], tour_arr[(k+1)%n]] for k in range(n))
            tours.append(tour_arr.copy())
            dists.append(dist)
            if dist < best_dist:
                best_dist = dist
                best_tour = tour_arr.copy()
                report_best_tour(best_tour)
                no_improve = 0
        # Evaporation
        pheromone *= (1 - rho)
        # Deposit from ants
        for tour_arr, dist in zip(tours, dists):
            for k in range(n):
                i, j = tour_arr[k], tour_arr[(k+1)%n]
                pheromone[i, j] += Q / dist
                pheromone[j, i] += Q / dist
        # Elitist reinforcement
        for k in range(n):
            i, j = best_tour[k], best_tour[(k+1)%n]
            pheromone[i, j] += Q / best_dist
            pheromone[j, i] += Q / best_dist
        # Stagnation check
        no_improve += 1
        if no_improve >= max_no_improve:
            pheromone = np.full((n, n), tau0)
            np.fill_diagonal(pheromone, 0)
            # Adaptive ILS: up to 2 perturbations
            for _ in range(2):
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
                    break
            no_improve = 0
    return best_tour