def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    rng = np.random.RandomState(42)
    best_tour = None
    best_cost = float('inf')
    for _ in range(10):
        tour = np.arange(n)
        visited = np.zeros(n, dtype=bool)
        start = rng.randint(n)
        tour[0] = start
        visited[start] = True
        for i in range(1, n):
            last = tour[i-1]
            dists = distance_matrix[last].copy()
            dists[visited] = np.inf
            k = max(1, int(n * 0.3))
            nearest = np.argsort(dists)[:k]
            next_city = rng.choice(nearest)
            tour[i] = next_city
            visited[next_city] = True
        improved = True
        while improved:
            improved = False
            for i in range(n-1):
                for j in range(i+2, n):
                    a = distance_matrix[tour[i], tour[i+1]]
                    b = distance_matrix[tour[j], tour[(j+1)%n]]
                    c = distance_matrix[tour[i], tour[j]]
                    d = distance_matrix[tour[i+1], tour[(j+1)%n]]
                    if c + d < a + b:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
        cost = distance_matrix[tour[-1], tour[0]]
        for k in range(n-1):
            cost += distance_matrix[tour[k], tour[k+1]]
        if cost < best_cost:
            best_cost = cost
            best_tour = tour.copy()
            report_best_tour(best_tour)
    return best_tour