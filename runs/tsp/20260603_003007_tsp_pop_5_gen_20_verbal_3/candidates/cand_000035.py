import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour

    num_ants = min(20, n)
    max_iter = 50
    alpha = 1.0
    beta = 2.0
    rho = 0.5
    q0 = 0.9
    tau0 = 1e-6

    tau = np.full((n, n), tau0)
    eta = 1.0 / (distance_matrix + np.eye(n))
    np.fill_diagonal(eta, 0.0)

    best_tour = None
    best_dist = np.inf

    for iteration in range(max_iter):
        tours = []
        for ant in range(num_ants):
            start = np.random.randint(n)
            tour = [start]
            visited = {start}
            for _ in range(n - 1):
                i = tour[-1]
                unvisited = [j for j in range(n) if j not in visited]
                if not unvisited:
                    break
                phe = tau[i, unvisited] ** alpha
                heu = eta[i, unvisited] ** beta
                probs = phe * heu
                probs = probs / probs.sum()
                if np.random.rand() < q0:
                    next_city = unvisited[np.argmax(probs)]
                else:
                    next_city = np.random.choice(unvisited, p=probs)
                tour.append(next_city)
                visited.add(next_city)
            tours.append(np.array(tour, dtype=np.int32))

        for idx, tour in enumerate(tours):
            improved = True
            while improved:
                improved = False
                n_len = n
                for i in range(n_len):
                    i1 = (i + 1) % n_len
                    for j in range(i + 2, n_len):
                        j1 = (j + 1) % n_len
                        delta = (distance_matrix[tour[i], tour[j]] +
                                 distance_matrix[tour[i1], tour[j1]] -
                                 distance_matrix[tour[i], tour[i1]] -
                                 distance_matrix[tour[j], tour[j1]])
                        if delta < -1e-12:
                            tour[i+1:j+1] = tour[i+1:j+1][::-1]
                            improved = True
                            break
                    if improved:
                        break
            tours[idx] = tour

        for tour in tours:
            dist = 0.0
            for k in range(n):
                dist += distance_matrix[tour[k], tour[(k+1)%n]]
            if dist < best_dist - 1e-12:
                best_dist = dist
                best_tour = tour.copy()
                report_best_tour(best_tour)

        tau *= (1 - rho)
        if best_tour is not None:
            delta_tau = 1.0 / best_dist
            for k in range(n):
                i = best_tour[k]
                j = best_tour[(k+1)%n]
                tau[i, j] += delta_tau
                tau[j, i] += delta_tau

    return best_tour