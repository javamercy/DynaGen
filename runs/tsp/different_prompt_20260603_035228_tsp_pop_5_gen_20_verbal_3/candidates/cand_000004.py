import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 0:
        return np.array([])
    if n == 1:
        tour = np.array([0])
        report_best_tour(tour)
        return tour
    # parameters
    n_ants = max(10, n)
    n_iter = max(100, n * 10)
    alpha = 1.0
    beta = 2.0
    evap = 0.5
    Q = 1.0
    # heuristic matrix (1/distance, avoid inf)
    heuristic = 1.0 / (distance_matrix + 1e-10)
    # initialize pheromone
    tau = np.ones((n, n)) * (1.0 / (n * np.mean(distance_matrix)))
    # best tour
    best_tour = None
    best_len = np.inf
    # early valid tour: construct one ant tour
    first_tour = np.zeros(n, dtype=int)
    first_tour[0] = np.random.randint(n)
    visited = {first_tour[0]}
    for i in range(1, n):
        last = first_tour[i-1]
        probs = heuristic[last] ** beta * tau[last] ** alpha
        probs[list(visited)] = 0
        probs /= probs.sum()
        next_city = np.random.choice(n, p=probs)
        first_tour[i] = next_city
        visited.add(next_city)
    first_len = sum(distance_matrix[first_tour[i], first_tour[(i+1)%n]] for i in range(n))
    report_best_tour(first_tour)
    best_tour = first_tour.copy()
    best_len = first_len
    for it in range(n_iter):
        tours = np.zeros((n_ants, n), dtype=int)
        lengths = np.zeros(n_ants)
        for ant in range(n_ants):
            start = np.random.randint(n)
            tours[ant, 0] = start
            visited = {start}
            for i in range(1, n):
                last = tours[ant, i-1]
                probs = heuristic[last] ** beta * tau[last] ** alpha
                probs[list(visited)] = 0
                # avoid zero sum
                if probs.sum() == 0:
                    # pick random unvisited
                    unvisited = [c for c in range(n) if c not in visited]
                    next_city = np.random.choice(unvisited)
                else:
                    probs /= probs.sum()
                    next_city = np.random.choice(n, p=probs)
                tours[ant, i] = next_city
                visited.add(next_city)
            # compute length
            lengths[ant] = sum(distance_matrix[tours[ant, i], tours[ant, (i+1)%n]] for i in range(n))
        # update best
        idx = np.argmin(lengths)
        if lengths[idx] < best_len:
            best_len = lengths[idx]
            best_tour = tours[idx].copy()
            report_best_tour(best_tour)
        # pheromone evaporation
        tau *= (1 - evap)
        # deposit on best tour
        delta = Q / best_len
        for i in range(n):
            a, b = best_tour[i], best_tour[(i+1)%n]
            tau[a, b] += delta
            tau[b, a] += delta
    return best_tour