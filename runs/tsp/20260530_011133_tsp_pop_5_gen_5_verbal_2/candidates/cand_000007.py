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
        # For each unvisited city, compute insertion costs
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
        # Regret: difference between second-best and best; tie-break by smaller best cost
        regret = {city: second_best_costs[city] - best_costs[city] for city in remaining}
        max_regret = max(regret.values())
        candidates = [city for city in remaining if regret[city] == max_regret]
        # Choose candidate with smallest best cost (tie-break)
        chosen = min(candidates, key=lambda x: best_costs[x])
        # Insert at best position
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
        n = len(tour)
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
        n = len(tour)
        while improved:
            improved = False
            for chain_len in [1, 2, 3]:  # chain lengths 1,2,3
                for start in range(n):
                    # Extract chain
                    chain = []
                    for k in range(chain_len):
                        chain.append(tour[(start + k) % n])
                    # Remove chain from tour
                    tour_list = tour.tolist()
                    for city in chain:
                        tour_list.remove(city)
                    # Try inserting chain at every position
                    new_tour_list = tour_list[:]
                    for pos in range(len(tour_list) + 1):
                        # Insert chain in original order
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

    # Apply 2-opt first
    best_tour, best_dist = two_opt(best_tour, best_dist)
    # Apply Or-opt
    best_tour, best_dist = or_opt(best_tour, best_dist)
    # Final 2-opt to polish
    best_tour, best_dist = two_opt(best_tour, best_dist)

    return best_tour