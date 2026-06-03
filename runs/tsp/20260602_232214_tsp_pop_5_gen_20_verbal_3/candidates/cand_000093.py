def solve_tsp(dm):
    n = dm.shape[0]
    if n == 1: return np.array([0])
    if n == 2: return np.array([0,1])
    def cost(t):
        return sum(dm[t[i], t[(i+1)%n]] for i in range(n))
    start = 0
    tour = [start]
    visited = {start}
    cur = start
    for _ in range(n-1):
        best_v = min((v for v in range(n) if v not in visited), key=lambda v: dm[cur,v])
        tour.append(best_v)
        visited.add(best_v)
        cur = best_v
    tour = np.array(tour)
    best_tour = tour.copy()
    best_cost = cost(tour)
    report_best_tour(tour)
    def improve(tour):
        nonlocal best_cost, best_tour
        curr = cost(tour)
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a,b,c,d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                    delta = dm[a,c] + dm[b,d] - dm[a,b] - dm[c,d]
                    if delta < -1e-12:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        curr += delta
                        improved = True
                        if curr < best_cost:
                            best_cost = curr
                            best_tour = tour.copy()
                            report_best_tour(best_tour)
                        break
                if improved:
                    break
        return tour, curr
    tour, _ = improve(tour)
    rng = np.random.default_rng()
    for _ in range(5):
        new_tour = rng.permutation(n)
        improve(new_tour)
    return best_tour