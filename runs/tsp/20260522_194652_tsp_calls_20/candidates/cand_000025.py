import numpy as np

def solve_tsp(distance_matrix, seed, budget):
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour)
        return tour

    rng = np.random.default_rng(seed)
    eps = 1e-12

    # Precompute candidate lists (nearest neighbors) for each city
    max_candidates = min(20, n-1) if n > 1 else 0
    candidate_lists = []
    for city in range(n):
        dists = distance_matrix[city]
        # sort indices by distance, ignoring self
        order = np.argsort(dists)
        order = order[order != city]
        candidate_lists.append(set(order[:max_candidates]))

    def tour_length(t):
        total = 0.0
        for i in range(n):
            total += distance_matrix[t[i], t[(i+1)%n]]
        return total

    def farthest_insertion():
        start = rng.integers(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            max_dist = -1
            farthest_candidates = []
            for city in unvisited:
                min_dist = min(distance_matrix[city, t] for t in tour)
                if min_dist > max_dist:
                    max_dist = min_dist
                    farthest_candidates = [city]
                elif min_dist == max_dist:
                    farthest_candidates.append(city)
            next_city = rng.choice(farthest_candidates)
            unvisited.remove(next_city)
            best_pos = -1
            best_increase = float('inf')
            for i in range(len(tour)):
                j = (i+1) % len(tour)
                increase = distance_matrix[tour[i], next_city] + distance_matrix[next_city, tour[j]] - distance_matrix[tour[i], tour[j]]
                if increase < best_increase:
                    best_increase = increase
                    best_pos = i+1
            tour.insert(best_pos, next_city)
        return np.array(tour, dtype=np.int32)

    # Initial tour
    tour = farthest_insertion()
    best_tour = tour.copy()
    best_len = tour_length(tour)
    report_best_tour(best_tour.copy())

    # Position array: maps city -> index in tour
    pos = np.empty(n, dtype=np.int32)
    for idx, city in enumerate(tour):
        pos[city] = idx

    while budget > 0:
        improved = False
        # Scan all i in order
        for i in range(n):
            if budget <= 0:
                break
            city_i = tour[i]
            # Consider candidates: neighbors of city_i and of next city
            next_i = (i+1) % n
            city_next = tour[next_i]
            neighbors = candidate_lists[city_i] | candidate_lists[city_next]
            # Remove self and adjacent cities? Actually we want j such that segment reversal is valid.
            for candidate in neighbors:
                if budget <= 0:
                    break
                j = pos[candidate]
                # Skip invalid moves: j must be > i? Actually 2-opt works with i < j and not adjacent.
                if j <= i or j == i+1 or (i == 0 and j == n-1):
                    continue
                # compute delta
                # edges removed: (i, i+1), (j, j+1)
                # edges added: (i, j), (i+1, j+1)
                a = tour[i]
                b = tour[(i+1)%n]
                c = tour[j]
                d = tour[(j+1)%n]
                delta = -distance_matrix[a, b] - distance_matrix[c, d] + distance_matrix[a, c] + distance_matrix[b, d]
                budget -= 1
                if delta < -eps:
                    # Apply 2-opt move: reverse segment (i+1 ... j)
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    # Update positions for affected cities
                    for idx in range(i+1, j+1):
                        pos[tour[idx]] = idx
                    improved = True
                    new_len = tour_length(tour)
                    if new_len < best_len - eps:
                        best_len = new_len
                        best_tour = tour.copy()
                        report_best_tour(best_tour.copy())
                    # First improvement: break out to restart scanning from beginning
                    break
            if improved:
                break
        if not improved and budget > 0:
            # Restart: decrement budget (counts as a step)
            budget -= 1
            tour = farthest_insertion()
            # Update positions
            for idx, city in enumerate(tour):
                pos[city] = idx
            new_len = tour_length(tour)
            if new_len < best_len - eps:
                best_len = new_len
                best_tour = tour.copy()
                report_best_tour(best_tour.copy())
        elif not improved:
            break
    return best_tour