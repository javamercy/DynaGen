import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=int)
        report_best_tour(tour)
        return tour

    def total_distance(tour):
        d = 0.0
        for i in range(n-1):
            d += distance_matrix[tour[i], tour[i+1]]
        d += distance_matrix[tour[-1], tour[0]]
        return d

    # Regret insertion construction with tie-breaking
    start = 0
    farthest = np.argmax(distance_matrix[start])
    tour = [start, farthest]
    remaining = set(range(n)) - {start, farthest}
    while remaining:
        best_costs = {}
        second_best_costs = {}
        for city in remaining:
            costs = []
            for i in range(len(tour)):
                prev = tour[i]
                nxt = tour[(i+1) % len(tour)]
                cost = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
                costs.append(cost)
            costs.sort()
            best_costs[city] = costs[0]
            second_best_costs[city] = costs[1] if len(costs) > 1 else costs[0]
        regret = {city: second_best_costs[city] - best_costs[city] for city in remaining}
        max_regret = max(regret.values())
        candidates = [city for city in remaining if regret[city] == max_regret]
        chosen = min(candidates, key=lambda x: best_costs[x])
        best_cost = float('inf')
        best_idx = -1
        for i in range(len(tour)):
            prev = tour[i]
            nxt = tour[(i+1) % len(tour)]
            cost = distance_matrix[prev, chosen] + distance_matrix[chosen, nxt] - distance_matrix[prev, nxt]
            if cost < best_cost:
                best_cost = cost
                best_idx = i + 1
        tour.insert(best_idx, chosen)
        remaining.remove(chosen)
    tour = np.array(tour, dtype=int)
    best_tour = tour.copy()
    best_dist = total_distance(best_tour)
    report_best_tour(best_tour)

    # Local search: 2-opt and Or-opt
    def two_opt(tour, dist):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    if j - i == 1:
                        continue
                    new_tour = np.concatenate([tour[:i+1], tour[j:i:-1], tour[j+1:]])
                    new_dist = total_distance(new_tour)
                    if new_dist < dist - 1e-12:
                        tour = new_tour
                        dist = new_dist
                        improved = True
                        report_best_tour(tour)
        return tour, dist

    def or_opt(tour, dist):
        improved = True
        while improved:
            improved = False
            for chain_len in [1, 2, 3]:
                for start in range(n):
                    chain = [tour[(start + k) % n] for k in range(chain_len)]
                    tour_list = tour.tolist()
                    for city in chain:
                        tour_list.remove(city)
                    for pos in range(len(tour_list) + 1):
                        cand = tour_list[:pos] + chain + tour_list[pos:]
                        cand = np.array(cand, dtype=int)
                        cand_dist = total_distance(cand)
                        if cand_dist < dist - 1e-12:
                            tour = cand
                            dist = cand_dist
                            improved = True
                            report_best_tour(tour)
                            break
                    if improved:
                        break
                if improved:
                    break
        return tour, dist

    def double_bridge(tour):
        n = len(tour)
        # choose four distinct indices, sorted
        indices = sorted(np.random.choice(range(1, n-1), 4, replace=False))  # exclude endpoints?
        a, b, c, d = indices
        # double bridge: reorder segments
        new_tour = np.concatenate([tour[:a], tour[c:d], tour[b:c], tour[a:b], tour[d:]])
        return new_tour

    # Main loop with restarts
    max_restarts = 5
    for restart in range(max_restarts):
        # Local search intensification
        best_tour, best_dist = two_opt(best_tour, best_dist)
        best_tour, best_dist = or_opt(best_tour, best_dist)
        best_tour, best_dist = two_opt(best_tour, best_dist)
        if restart == max_restarts - 1:
            break
        # Perturbation
        perturbed = double_bridge(best_tour)
        perturbed_dist = total_distance(perturbed)
        # If worse, local search will try to improve; if better, update
        if perturbed_dist < best_dist:
            best_tour = perturbed
            best_dist = perturbed_dist
            report_best_tour(best_tour)
        else:
            # Accept perturbation even if worse to escape local optima
            best_tour = perturbed
            best_dist = perturbed_dist
        # Continue loop

    return best_tour