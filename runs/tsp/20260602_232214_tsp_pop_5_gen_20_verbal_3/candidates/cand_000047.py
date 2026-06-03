import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def compute_dist(tour):
        s = 0
        for i in range(n):
            s += distance_matrix[tour[i], tour[(i+1)%n]]
        return s

    def two_opt(tour):
        best = tour.copy()
        best_dist = compute_dist(best)
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a = best[i]; b = best[i+1]; c = best[j]; d = best[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-10:
                        new = best.copy()
                        new[i+1:j+1] = best[j:i:-1]
                        best_dist += delta
                        best = new
                        improved = True
                        report_best_tour(best)
        return best, best_dist

    def nearest_neighbor(start):
        tour = [start]
        visited = {start}
        cur = start
        for _ in range(n-1):
            min_dist = np.inf
            next_node = None
            for v in range(n):
                if v not in visited:
                    d = distance_matrix[cur, v]
                    if d < min_dist:
                        min_dist = d
                        next_node = v
            tour.append(next_node)
            visited.add(next_node)
            cur = next_node
        return np.array(tour)

    def regret_insertion(start):
        tour = [start]
        visited = {start}
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            best_cost = np.inf
            best_node = None
            best_pos = None
            for u in unvisited:
                min_inc = np.inf
                min_pos = None
                for k in range(len(tour)):
                    a = tour[k]
                    b = tour[(k+1)%len(tour)]
                    inc = distance_matrix[a,u] + distance_matrix[u,b] - distance_matrix[a,b]
                    if inc < min_inc:
                        min_inc = inc
                        min_pos = k+1
                if min_inc < best_cost:
                    best_cost = min_inc
                    best_node = u
                    best_pos = min_pos
            tour.insert(best_pos, best_node)
            visited.add(best_node)
            unvisited.remove(best_node)
        return np.array(tour)

    best_tour = None
    best_dist = float('inf')
    for _ in range(5):
        if np.random.rand() < 0.5:
            start = np.random.randint(n)
            tour = nearest_neighbor(start)
        else:
            start = np.random.randint(n)
            tour = regret_insertion(start)
        dist = compute_dist(tour)
        if dist < best_dist:
            best_tour = tour.copy()
            best_dist = dist
            report_best_tour(best_tour)
        improved_tour, improved_dist = two_opt(tour)
        if improved_dist < best_dist:
            best_tour = improved_tour.copy()
            best_dist = improved_dist
            report_best_tour(best_tour)
    return best_tour