import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    
    def nn_tour(start):
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        cur = start
        while unvisited:
            nxt = min(unvisited, key=lambda x: distance_matrix[cur, x])
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        return np.array(tour)
    
    def perturbed_nn():
        start = np.random.randint(n)
        tour = nn_tour(start)
        # apply 2 random 2-opt swaps
        for _ in range(2):
            i = np.random.randint(0, n-1)
            j = np.random.randint(i+2, n)
            if j - i == 1:
                continue
            # avoid degenerate
            tour[i+1:j+1] = tour[i+1:j+1][::-1]
        return tour
    
    def tour_distance(tour):
        return sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    
    # initial global best from NN start 0
    global_best_tour = nn_tour(0)
    global_best_dist = tour_distance(global_best_tour)
    report_best_tour(global_best_tour)
    
    # phase best (used for stagnation)
    phase_best_tour = global_best_tour.copy()
    phase_best_dist = global_best_dist
    
    # pheromone initialization
    mean_dist = np.mean(distance_matrix[distance_matrix > 0])
    tau0 = 1.0 / (n * mean_dist)
    pheromone = np.full((n, n), tau0)
    np.fill_diagonal(pheromone, 0)
    
    # parameters
    n_ants = 25
    n_iterations = 120
    beta = 2.0
    Q = 1.0
    eta = np.divide(1.0, distance_matrix + np.eye(n), out=np.zeros_like(distance_matrix), where=distance_matrix+np.eye(n)!=0)
    np.fill_diagonal(eta, 0)
    
    stagnation = 0
    max_stagnation = 30
    
    for it in range(n_iterations):
        progress = it / n_iterations
        rho = max(0.2, 0.7 - 0.4 * progress)
        alpha = 0.5 + 1.0 * progress
        
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
                    nxt = np.random.choice(choices)
                else:
                    prob = prob / prob_sum
                    nxt = np.random.choice(n, p=prob)
                tour.append(nxt)
                visited.add(nxt)
            
            # 2-opt local search
            improved = True
            tour_arr = np.array(tour)
            while improved:
                improved = False
                for i in range(n-1):
                    for j in range(i+2, n):
                        a, b = tour_arr[i], tour_arr[i+1]
                        c, d = tour_arr[j], tour_arr[(j+1)%n]
                        if distance_matrix[a, c] + distance_matrix[b, d] < distance_matrix[a, b] + distance_matrix[c, d]:
                            tour_arr[i+1:j+1] = tour_arr[i+1:j+1][::-1]
                            improved = True
            
            dist = tour_distance(tour_arr)
            if dist < global_best_dist:
                global_best_dist = dist
                global_best_tour = tour_arr.copy()
                report_best_tour(global_best_tour)
            if dist < phase_best_dist:
                phase_best_dist = dist
                phase_best_tour = tour_arr.copy()
            
            # deposit pheromone
            for k in range(n):
                i, j = tour_arr[k], tour_arr[(k+1)%n]
                pheromone[i, j] += Q / dist
                pheromone[j, i] += Q / dist
        
        # evaporation
        pheromone *= (1 - rho)
        
        # stagnation check on phase best
        if phase_best_dist == prev_phase_best if it > 0 else False:
            stagnation += 1
        else:
            stagnation = 0
        if it == 0:
            prev_phase_best = phase_best_dist
        else:
            prev_phase_best = phase_best_dist
        
        if stagnation >= max_stagnation:
            # restart: reset pheromone and set phase best to perturbed NN
            pheromone = np.full((n, n), tau0)
            np.fill_diagonal(pheromone, 0)
            perturbed_tour = perturbed_nn()
            perturbed_dist = tour_distance(perturbed_tour)
            if perturbed_dist < global_best_dist:
                global_best_dist = perturbed_dist
                global_best_tour = perturbed_tour.copy()
                report_best_tour(global_best_tour)
            phase_best_tour = perturbed_tour.copy()
            phase_best_dist = perturbed_dist
            stagnation = 0
            prev_phase_best = phase_best_dist
    
    return global_best_tour