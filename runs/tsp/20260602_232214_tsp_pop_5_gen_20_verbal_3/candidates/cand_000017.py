import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    best_tour = None
    best_dist = float('inf')
    max_iters = max(10, n // 10)
    for _ in range(max_iters):
        unvisited = set(range(n))
        start = np.random.randint(n)
        tour = [start]
        unvisited.remove(start)
        while unvisited:
            best_candidates = []
            best_cost = float('inf')
            for city in unvisited:
                min_cost = float('inf')
                pos = 0
                for k in range(len(tour)):
                    a = tour[k]
                    b = tour[(k+1) % len(tour)]
                    cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                    if cost < min_cost:
                        min_cost = cost
                        pos = k + 1
                if min_cost < best_cost:
                    best_cost = min_cost
                    best_candidates = [(city, pos, min_cost)]
                elif min_cost == best_cost:
                    best_candidates.append((city, pos, min_cost))
            k = min(3, len(best_candidates))
            idx = np.random.choice(len(best_candidates), 1)[0]
            city, pos, _ = best_candidates[idx]
            tour.insert(pos, city)
            unvisited.remove(city)
        tour_arr = np.array(tour)
        dist = sum(distance_matrix[tour_arr[i], tour_arr[(i+1)%n]] for i in range(n))
        if dist < best_dist:
            best_dist = dist
            best_tour = tour_arr.copy()
            report_best_tour(best_tour)
        # 2-opt local search
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a = best_tour[i]
                    b = best_tour[i+1]
                    c = best_tour[j]
                    d = best_tour[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-10:
                        new_tour = best_tour.copy()
                        new_tour[i+1:j+1] = best_tour[j:i:-1]
                        best_tour = new_tour
                        best_dist += delta
                        improved = True
                        report_best_tour(best_tour)
        # Perturbation: random 2-opt move that worsens tour
        if n > 4:
            perturbed = False
            for _ in range(10):
                i = np.random.randint(n-2)
                j = np.random.randint(i+2, n)
                a = best_tour[i]
                b = best_tour[i+1]
                c = best_tour[j]
                d = best_tour[(j+1)%n]
                delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                if delta > 0:
                    new_tour = best_tour.copy()
                    new_tour[i+1:j+1] = best_tour[j:i:-1]
                    best_tour = new_tour
                    best_dist += delta
                    perturbed = True
                    break
            if not perturbed:
                # if no worsening found, swap two random cities
                i, j = np.random.choice(n, 2, replace=False)
                best_tour[i], best_tour[j] = best_tour[j], best_tour[i]
                best_dist = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n]] for i in range(n))
    return best_tour