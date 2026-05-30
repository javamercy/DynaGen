import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=int)

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

    # Local search: 2-opt
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

    # Or-opt: chain lengths 1,2,3
    def or_opt(tour, dist):
        improved = True
        while improved:
            improved = False
            for chain_len in [1, 2, 3]:
                for start_idx in range(n):
                    chain = [tour[(start_idx + k) % n] for k in range(chain_len)]
                    tour_list = tour.tolist()
                    for city in chain:
                        tour_list.remove(city)
                    for pos in range(len(tour_list) + 1):
                        cand = np.array(tour_list[:pos] + chain + tour_list[pos:], dtype=int)
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

    # Double-bridge perturbation
    def double_bridge(tour, dist):
        n = len(tour)
        # Choose three distinct positions in [1, n-2] to split into four segments
        a = np.random.randint(1, n // 2)
        b = np.random.randint(a + 1, n - 2)
        c = np.random.randint(b + 1, n - 1)
        # Segments: [0:a], [a:b], [b:c], [c:]
        seg1 = tour[:a]
        seg2 = tour[a:b]
        seg3 = tour[b:c]
        seg4 = tour[c:]
        # Reorder as seg1, seg3, seg2, seg4
        new_tour = np.concatenate([seg1, seg3, seg2, seg4])
        new_dist = total_distance(new_tour)
        return new_tour, new_dist

    # Iterative improvement with restarts
    max_restarts = 5
    for _ in range(max_restarts):
        # Apply local search
        best_tour, best_dist = two_opt(best_tour, best_dist)
        best_tour, best_dist = or_opt(best_tour, best_dist)
        best_tour, best_dist = two_opt(best_tour, best_dist)
        # Perturb if not last iteration
        if _ < max_restarts - 1:
            pert_tour, pert_dist = double_bridge(best_tour, best_dist)
            if pert_dist < best_dist:
                best_tour = pert_tour
                best_dist = pert_dist
                report_best_tour(best_tour)
            else:
                best_tour = pert_tour
                best_dist = pert_dist

    return best_tour