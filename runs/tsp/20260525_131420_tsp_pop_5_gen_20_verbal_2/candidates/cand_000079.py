import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour.copy())
        return tour
    rng = np.random.RandomState(seed)
    # Precompute nearest neighbor lists (size 50)
    nn_lists = [np.argsort(distance_matrix[i])[1:51] for i in range(n)]
    
    def regret_insertion():
        start = rng.randint(n)
        dists = distance_matrix[start].copy()
        dists[start] = np.inf
        second = np.argmin(dists)
        tour = [start, second]
        remaining = set(range(n)) - {start, second}
        while remaining:
            best_regret = -1e100
            best_city = None
            best_pos = None
            best_cost = None
            for city in remaining:
                L = len(tour)
                costs = [distance_matrix[tour[i], city] + distance_matrix[city, tour[(i+1)%L]] - distance_matrix[tour[i], tour[(i+1)%L]] for i in range(L)]
                sorted_costs = sorted(costs)
                b = sorted_costs[0]
                second_best = sorted_costs[1] if len(sorted_costs) > 1 else b
                regret = second_best - b
                if regret > best_regret or (regret == best_regret and (best_cost is None or b < best_cost)):
                    best_regret = regret
                    best_city = city
                    best_pos = int(np.argmin(costs))
                    best_cost = b
            insert_idx = best_pos + 1
            tour = tour[:insert_idx] + [best_city] + tour[insert_idx:]
            remaining.remove(best_city)
        return np.array(tour, dtype=np.int32)
    
    # Construction: 10 restarts
    best_tour = None
    best_length = np.inf
    for _ in range(10):
        t = regret_insertion()
        l = sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))
        if l < best_length:
            best_length = l
            best_tour = t.copy()
    report_best_tour(best_tour.copy())
    
    current_tour = best_tour.copy()
    current_length = best_length
    attempts = 0
    max_attempts = budget
    stagnation = 0
    
    while attempts < max_attempts:
        # Steepest-ascent 2-opt with candidate list
        best_delta = 0
        best_move = None
        pos = {city: idx for idx, city in enumerate(current_tour)}
        tour_len = n
        for i in range(tour_len - 1):
            if attempts >= max_attempts:
                break
            a = current_tour[i]
            b = current_tour[(i+1) % tour_len]
            # candidates: neighbors of a or b? We'll use neighbors of a for simplicity
            candidates = nn_lists[a]
            for c in candidates:
                j = pos[c]
                if j < 0 or j >= tour_len:
                    continue
                if j <= i+1 or j >= tour_len - 1:
                    continue
                # Consider move (i, j)
                d = current_tour[(j+1) % tour_len]
                old = distance_matrix[a, b] + distance_matrix[c, d]
                new = distance_matrix[a, c] + distance_matrix[b, d]
                attempts += 1
                if attempts > max_attempts:
                    break
                if new < old:
                    delta = new - old
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, j)
            if attempts >= max_attempts:
                break
        if best_move is not None and best_delta < 0:
            i, j = best_move
            # reverse segment i+1..j
            current_tour[i+1:j+1] = current_tour[i+1:j+1][::-1]
            current_length += best_delta
            report_best_tour(current_tour.copy())
            if current_length < best_length:
                best_length = current_length
                best_tour = current_tour.copy()
            stagnation = 0
        else:
            # No improvement: apply random 2-opt swap (perturbation)
            stagnation += 1
            if stagnation >= 5 and attempts < max_attempts:
                # restart from new construction
                t = regret_insertion()
                l = sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))
                current_tour = t.copy()
                current_length = l
                reports if l < best_length?
                if current_length < best_length:
                    best_length = current_length
                    best_tour = current_tour.copy()
                    report_best_tour(best_tour.copy())
                stagnation = 0
                continue
            # random valid 2-opt move
            found = False
            for _ in range(100):
                i = rng.randint(0, tour_len - 2)
                j = rng.randint(i+2, tour_len)
                if i == 0 and j == tour_len - 1:
                    continue
                # apply
                current_tour[i+1:j+1] = current_tour[i+1:j+1][::-1]
                # recalc length
                current_length = sum(distance_matrix[current_tour[k], current_tour[(k+1)%tour_len]] for k in range(tour_len))
                attempts += 1
                if attempts > max_attempts:
                    break
                found = True
                break
            if not found:
                break
            # report if new best
            if current_length < best_length:
                best_length = current_length
                best_tour = current_tour.copy()
                report_best_tour(best_tour.copy())
        if attempts >= max_attempts:
            break
    return best_tour