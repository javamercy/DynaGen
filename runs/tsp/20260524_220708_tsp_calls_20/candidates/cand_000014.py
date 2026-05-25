import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def cheapest_insertion(seed):
        np.random.seed(seed)
        perm = np.random.permutation(n)
        tour = [perm[0], perm[1]]
        unvisited = set(perm[2:])
        while unvisited:
            best_inc = float('inf')
            candidates = []
            for city in unvisited:
                for i in range(len(tour)):
                    prev = tour[i]
                    nxt = tour[(i+1) % len(tour)]
                    inc = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
                    if inc < best_inc:
                        best_inc = inc
                        candidates = [(city, i)]
                    elif inc == best_inc:
                        candidates.append((city, i))
            idx = np.random.randint(len(candidates))
            city, pos = candidates[idx]
            tour.insert(pos+1, city)
            unvisited.remove(city)
        return np.array(tour)

    def farthest_insertion(seed):
        rng = np.random.RandomState(seed)
        tour = [rng.randint(n)]
        unvisited = set(range(n)) - {tour[0]}
        while unvisited:
            farthest = None
            max_min_dist = -1.0
            for city in unvisited:
                min_dist = np.min(distance_matrix[city, tour])
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    farthest = city
            best_pos = None
            best_increase = np.inf
            for i in range(len(tour)):
                prev = tour[i]
                nxt = tour[(i + 1) % len(tour)]
                inc = distance_matrix[prev, farthest] + distance_matrix[farthest, nxt] - distance_matrix[prev, nxt]
                if inc < best_increase:
                    best_increase = inc
                    best_pos = i + 1
            tour.insert(best_pos, farthest)
            unvisited.remove(farthest)
        return np.array(tour)

    tour1 = cheapest_insertion(seed)
    tour2 = farthest_insertion(seed + 1)

    def tour_distance(tour):
        d = 0.0
        for i in range(n):
            d += distance_matrix[tour[i], tour[(i+1) % n]]
        return d

    d1 = tour_distance(tour1)
    d2 = tour_distance(tour2)
    tour = tour1 if d1 <= d2 else tour2
    best_tour = tour.copy()
    best_dist = tour_distance(best_tour)
    report_best_tour(best_tour)

    if budget == 0:
        return best_tour

    # 2-opt improvement
    improved = True
    ops = 0
    while improved and ops < budget:
        improved = False
        for i in range(n-1):
            for j in range(i+2, n):
                if ops >= budget:
                    break
                i_next = (i+1) % n
                j_next = (j+1) % n
                old = distance_matrix[tour[i], tour[i_next]] + distance_matrix[tour[j], tour[j_next]]
                new = distance_matrix[tour[i], tour[j]] + distance_matrix[tour[i_next], tour[j_next]]
                if new < old:
                    tour[i_next:j+1] = tour[i_next:j+1][::-1]
                    improved = True
                    ops += 1
                    new_dist = tour_distance(tour)
                    if new_dist < best_dist:
                        best_dist = new_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break
            if improved:
                break
    return best_tour