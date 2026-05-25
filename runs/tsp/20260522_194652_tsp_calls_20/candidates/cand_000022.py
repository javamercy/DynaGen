import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour)
        return tour

    rng = np.random.default_rng(seed)

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
            # find farthest city from current tour
            max_dist = -1
            farthest_candidates = []
            for city in unvisited:
                # distance to nearest tour city
                min_dist = min(distance_matrix[city, t] for t in tour)
                if min_dist > max_dist:
                    max_dist = min_dist
                    farthest_candidates = [city]
                elif min_dist == max_dist:
                    farthest_candidates.append(city)
            # random tie-breaking
            next_city = rng.choice(farthest_candidates)
            unvisited.remove(next_city)
            # best insertion
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

    # initial tour
    tour = farthest_insertion()
    best_tour = tour.copy()
    best_len = tour_length(tour)
    report_best_tour(best_tour.copy())

    eps = 1e-12
    no_improve_streak = 0
    max_attempts = n * 10  # attempts before restart

    while budget > 0:
        # random swap local search
        improved = False
        for _ in range(max_attempts):
            if budget <= 0:
                break
            # pick two distinct random indices
            i = rng.integers(n)
            j = rng.integers(n)
            while j == i:
                j = rng.integers(n)
            if i > j:
                i, j = j, i
            # compute delta
            # edges removed: (i-1,i), (i,i+1), (j-1,j), (j,j+1)
            # edges added: (i-1,j), (j,i+1), (j-1,i), (i,j+1)
            # careful with wrap-around
            i_minus = (i-1) % n
            i_plus = (i+1) % n
            j_minus = (j-1) % n
            j_plus = (j+1) % n
            # if positions are adjacent, special cases
            if j == i+1 or (i == n-1 and j == 0):
                # adjacent swap: i and i+1
                delta = distance_matrix[tour[i_minus], tour[j]] + distance_matrix[tour[j], tour[i]] + distance_matrix[tour[i], tour[j_plus]] - \
                        (distance_matrix[tour[i_minus], tour[i]] + distance_matrix[tour[i], tour[j]] + distance_matrix[tour[j], tour[j_plus]])
            else:
                delta = distance_matrix[tour[i_minus], tour[j]] + distance_matrix[tour[j], tour[i_plus]] + \
                        distance_matrix[tour[j_minus], tour[i]] + distance_matrix[tour[i], tour[j_plus]] - \
                        (distance_matrix[tour[i_minus], tour[i]] + distance_matrix[tour[i], tour[i_plus]] + \
                         distance_matrix[tour[j_minus], tour[j]] + distance_matrix[tour[j], tour[j_plus]])
            budget -= 1
            if delta < -eps:
                # swap
                tour[i], tour[j] = tour[j], tour[i]
                improved = True
                new_len = tour_length(tour)
                if new_len < best_len - eps:
                    best_len = new_len
                    best_tour = tour.copy()
                    report_best_tour(best_tour.copy())
                break  # first improvement
        if not improved and budget > 0:
            # restart
            budget -= 1
            tour = farthest_insertion()
            new_len = tour_length(tour)
            if new_len < best_len - eps:
                best_len = new_len
                best_tour = tour.copy()
                report_best_tour(best_tour.copy())
            no_improve_streak = 0
        elif not improved:
            break
    return best_tour