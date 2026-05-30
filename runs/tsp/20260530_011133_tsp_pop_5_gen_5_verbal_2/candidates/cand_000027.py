import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=int)

    def total_distance(tour):
        d = 0.0
        for i in range(n - 1):
            d += distance_matrix[tour[i], tour[i + 1]]
        d += distance_matrix[tour[-1], tour[0]]
        return d

    def two_opt(tour, dist, max_passes=2):
        for _ in range(max_passes):
            improved = False
            for i in range(n):
                for j in range(i + 2, n):
                    if j - i == 1:
                        continue
                    new_tour = np.concatenate([tour[:i+1], tour[j:i:-1], tour[j+1:]])
                    new_dist = total_distance(new_tour)
                    if new_dist < dist - 1e-12:
                        tour = new_tour
                        dist = new_dist
                        improved = True
            if not improved:
                break
        return tour, dist

    def or_opt(tour, dist, max_passes=2):
        for _ in range(max_passes):
            improved = False
            for chain_len in [1, 2, 3]:
                for start in range(n):
                    chain = [tour[(start + k) % n] for k in range(chain_len)]
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
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return tour, dist

    def double_bridge(tour):
        n_len = len(tour)
        pos1 = np.random.randint(1, n_len // 3)
        pos2 = np.random.randint(pos1 + 1, 2 * n_len // 3)
        pos3 = np.random.randint(pos2 + 1, n_len - 1)
        A = tour[:pos1]
        B = tour[pos1:pos2]
        C = tour[pos2:pos3]
        D = tour[pos3:]
        return np.concatenate([A, C, B, D])

    # Regret insertion with tie-breaking (smallest best cost)
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
                nxt = tour[(i + 1) % len(tour)]
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
            nxt = tour[(i + 1) % len(tour)]
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

    # Initial local search
    tour, dist = two_opt(tour, best_dist)
    tour, dist = or_opt(tour, dist)
    tour, dist = two_opt(tour, dist)
    if dist < best_dist - 1e-12:
        best_tour = tour.copy()
        best_dist = dist
        report_best_tour(best_tour)

    # Restart loop
    for _ in range(2):
        perturbed = double_bridge(best_tour.copy())
        perturbed_dist = total_distance(perturbed)
        tour, dist = two_opt(perturbed, perturbed_dist)
        tour, dist = or_opt(tour, dist)
        tour, dist = two_opt(tour, dist)
        if dist < best_dist - 1e-12:
            best_tour = tour.copy()
            best_dist = dist
            report_best_tour(best_tour)

    return best_tour