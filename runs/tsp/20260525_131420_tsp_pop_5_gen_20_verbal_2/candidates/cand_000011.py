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
    steps_used = 0
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
        steps_used += 1
    # Report initial tour
    report_best_tour(tour)
    best_tour = tour[:]
    best_dist = distance_matrix[tour[-1], tour[0]]
    for i in range(len(tour)-1):
        best_dist += distance_matrix[tour[i], tour[i+1]]
    current_tour = tour[:]
    current_dist = best_dist
    # Improvement: 2-opt with remaining budget
    remaining_steps = budget - steps_used
    if remaining_steps < 0:
        remaining_steps = 0
    improved = True
    # For step counting, we'll count each 2-opt evaluation (i,j pair) as one step.
    # We'll limit number of pairs per pass to avoid excessive non-improving steps.
    # Use a fixed number of pairs per pass proportional to n.
    max_pairs_per_pass = 100 * n  # randomize a subset
    while remaining_steps > 0 and improved:
        improved = False
        n_tour = len(current_tour)
        # Generate random subset of pairs
        num_pairs = min(max_pairs_per_pass, remaining_steps)
        # Instead, generate all pairs and shuffle, but limit evaluation
        pairs = [(i, j) for i in range(n_tour) for j in range(i+2, n_tour)]
        rng.shuffle(pairs)
        evaluated = 0
        for i, j in pairs:
            if evaluated >= remaining_steps:
                break
            evaluated += 1
            a = current_tour[i]
            b = current_tour[(i+1) % n_tour]
            c = current_tour[j]
            d = current_tour[(j+1) % n_tour]
            delta = (distance_matrix[a, c] + distance_matrix[b, d]) - (distance_matrix[a, b] + distance_matrix[c, d])
            if delta < 0:
                current_tour = current_tour[:i+1] + current_tour[i+1:j+1][::-1] + current_tour[j+1:]
                current_dist += delta
                improved = True
                remaining_steps -= evaluated  # subtract evaluations used so far
                if current_dist < best_dist:
                    best_dist = current_dist
                    best_tour = current_tour[:]
                    report_best_tour(best_tour)
                break
        if not improved:
            # Double-bridge perturbation
            if remaining_steps > 0:
                # Apply random double-bridge move
                n_tour = len(current_tour)
                # Pick four random indices (distinct, properly ordered)
                indices = sorted(rng.choice(n_tour, size=4, replace=False))
                a, b, c, d = indices
                # Convert tour to list for slicing
                segments = [current_tour[:a], current_tour[a:b], current_tour[b:c], current_tour[c:d], current_tour[d:]]
                # Reorder: segment 1, segment 3, segment 2, segment 4
                new_tour = segments[0] + segments[2] + segments[1] + segments[3] + segments[4]
                current_tour = new_tour
                # Recompute distance
                current_dist = distance_matrix[current_tour[-1], current_tour[0]]
                for i in range(len(current_tour)-1):
                    current_dist += distance_matrix[current_tour[i], current_tour[i+1]]
                improved = True  # allow another pass
                # Report if improved but double-bridge might not improve; still allow search
                remaining_steps -= 1  # count perturbation as one step
    return np.array(best_tour)