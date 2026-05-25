import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = list(range(n))
        report_best_tour(tour)
        return np.array(tour)
    rng = np.random.RandomState(seed)
    if budget < n:
        tour = list(range(n))
        rng.shuffle(tour)
        report_best_tour(tour)
        return np.array(tour)
    # Construction: regret-insertion
    start = rng.randint(n)
    dists = distance_matrix[start]
    second = np.argmin(dists)
    if second == start:
        second = (start + 1) % n
    tour = [start, second]
    remaining = set(range(n)) - {start, second}
    # Regret insertion (no budget limit here, we'll account after)
    while remaining:
        best_regret = -1
        best_city = None
        best_pos = None
        best_cost = None
        for city in remaining:
            costs = []
            cur_len = len(tour)
            for p in range(cur_len + 1):
                left = tour[p-1] if p > 0 else tour[-1]
                right = tour[p] if p < cur_len else tour[0]
                cost = (distance_matrix[left, city] + distance_matrix[city, right] -
                        distance_matrix[left, right])
                costs.append(cost)
            sorted_costs = sorted(costs)
            best = sorted_costs[0]
            second_best = sorted_costs[1] if len(sorted_costs) > 1 else best
            regret = second_best - best
            if regret > best_regret or (regret == best_regret and best < best_cost):
                best_regret = regret
                best_city = city
                best_pos = costs.index(best)
                best_cost = best
        tour = tour[:best_pos] + [best_city] + tour[best_pos:]
        remaining.remove(best_city)
    # Report initial tour
    report_best_tour(tour)
    best_tour = tour[:]
    best_dist = distance_matrix[tour[-1], tour[0]]
    for i in range(len(tour)-1):
        best_dist += distance_matrix[tour[i], tour[i+1]]
    current_tour = tour[:]
    current_dist = best_dist
    # Improvement: 2-opt with remaining budget
    steps_construction = n - 2  # number of insertions performed
    remaining_steps = budget - steps_construction
    if remaining_steps < 0:
        remaining_steps = 0
    improved = True
    while remaining_steps > 0 and improved:
        improved = False
        n_tour = len(current_tour)
        # Generate list of all valid 2-opt pairs (i, j) with i < j-1
        pairs = []
        for i in range(n_tour):
            for j in range(i+2, n_tour):  # ensure segment length >= 2
                pairs.append((i, j))
        rng.shuffle(pairs)
        for i, j in pairs:
            if remaining_steps <= 0:
                break
            remaining_steps -= 1
            a = current_tour[i]
            b = current_tour[(i+1) % n_tour]
            c = current_tour[j]
            d = current_tour[(j+1) % n_tour]
            # delta = new edges - old edges
            delta = (distance_matrix[a, c] + distance_matrix[b, d]) - (distance_matrix[a, b] + distance_matrix[c, d])
            if delta < 0:
                # Apply 2-opt move
                current_tour = current_tour[:i+1] + current_tour[i+1:j+1][::-1] + current_tour[j+1:]
                current_dist += delta
                improved = True
                if current_dist < best_dist:
                    best_dist = current_dist
                    best_tour = current_tour[:]
                    report_best_tour(best_tour)
                break  # restart shuffle after each improvement to avoid edge interactions
    # Return best tour found
    return np.array(best_tour)