import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    if budget < n:
        tour = list(range(n))
        rng.shuffle(tour)
        report_best_tour(tour)
        return np.array(tour)
    # Find pair with maximum distance
    max_dist = -1
    pair = (0, 1)
    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i][j] > max_dist:
                max_dist = distance_matrix[i][j]
                pair = (i, j)
    tour = list(pair)
    remaining = set(range(n)) - set(pair)
    steps_left = budget - n + 2
    insertion_steps = 0
    while remaining:
        if steps_left <= 0:
            tour.extend(sorted(remaining))
            break
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
        insertion_steps += 1
        steps_left -= 1
    best_tour = tour[:]
    best_dist = tour_length(tour, distance_matrix)
    report_best_tour(best_tour)
    # Local search with remaining budget
    local_budget = max(0, steps_left)
    if local_budget > 0:
        for _ in range(local_budget):
            i = rng.randint(0, n-1)
            j = rng.randint(i+1, n) if i < n-1 else i
            # ensure i < j
            if i >= j:
                i, j = j, i
            # compute delta for 2-opt move (reverse segment i+1..j)
            n_tour = len(tour)
            i_next = (i + 1) % n_tour
            j_next = (j + 1) % n_tour
            delta = (distance_matrix[tour[i], tour[j]] +
                     distance_matrix[tour[i_next], tour[j_next]] -
                     distance_matrix[tour[i], tour[i_next]] -
                     distance_matrix[tour[j], tour[j_next]])
            if delta < 0:
                # apply reversal
                tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                new_dist = best_dist + delta
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_tour = tour[:]
                    report_best_tour(best_tour)
    return np.array(best_tour)

def tour_length(tour, mat):
    n = len(tour)
    return sum(mat[tour[i], tour[(i+1)%n]] for i in range(n))