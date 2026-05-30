import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=int)

    def compute_total_distance(tour):
        d = 0.0
        for i in range(n-1):
            d += distance_matrix[tour[i], tour[i+1]]
        d += distance_matrix[tour[-1], tour[0]]
        return d

    # Nearest neighbor initial tour
    visited = [0]
    current = 0
    available = set(range(1, n))
    while available:
        next_city = min(available, key=lambda x: distance_matrix[current, x])
        visited.append(next_city)
        available.remove(next_city)
        current = next_city
    best_tour = np.array(visited)
    best_dist = compute_total_distance(best_tour)
    report_best_tour(best_tour)

    # Regret-insertion construction
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
        chosen = max(regret, key=lambda x: regret[x])
        best_cost = float('inf')
        best_idx = -1
        for i in range(len(tour)):
            prev = tour[i]
            nxt = tour[(i+1) % len(tour)]
            cost = distance_matrix[prev, chosen] + distance_matrix[chosen, nxt] - distance_matrix[prev, nxt]
            if cost < best_cost:
                best_cost = cost
                best_idx = i+1
        tour.insert(best_idx, chosen)
        remaining.remove(chosen)
    tour_regret = np.array(tour)
    dist_regret = compute_total_distance(tour_regret)
    if dist_regret < best_dist - 1e-12:
        best_dist = dist_regret
        best_tour = tour_regret.copy()
        report_best_tour(best_tour)

    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+2, n):
                new_tour = np.concatenate([best_tour[:i+1], best_tour[j:i:-1], best_tour[j+1:]])
                new_dist = compute_total_distance(new_tour)
                if new_dist < best_dist - 1e-12:
                    best_dist = new_dist
                    best_tour = new_tour
                    improved = True
                    report_best_tour(best_tour)
                    break
            if improved:
                break

    # 3-opt improvement (limited to 2 iterations)
    improved = True
    max_iter = 2
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        for i in range(n):
            for j in range(i+2, n):
                for k in range(j+2, n):
                    # Original tour: A[0:i+1], B[i+1:j+1], C[j+1:k+1], D[k+1:]
                    A = best_tour[:i+1]
                    B = best_tour[i+1:j+1]
                    C = best_tour[j+1:k+1]
                    D = best_tour[k+1:]
                    # Four reconnections
                    candidates = []
                    # 1: A, rev(B), C, D
                    candidates.append(np.concatenate([A, B[::-1], C, D]))
                    # 2: A, B, rev(C), D
                    candidates.append(np.concatenate([A, B, C[::-1], D]))
                    # 3: A, rev(B), rev(C), D
                    candidates.append(np.concatenate([A, B[::-1], C[::-1], D]))
                    # 4: A, rev(C), B, D (this is same as original order for rev of just B/C? Actually original is A,B,C,D. We also consider swapping? But typical 3-opt includes these three. We'll add one more: A, rev(C), B, D)
                    # Actually there are 4 possibilities: (A,B,C,D), (A,rev(B),C,D), (A,B,rev(C),D), (A,rev(B),rev(C),D). We already have that. The fourth is actually (A,rev(C),B,D)? That's a different orientation. But careful: reversing both B and C is (A, rev(C), rev(B), D) if we reverse order as well? This is complex. We'll stick to the three reversals plus original? But original is handled by 2-opt. Let's implement the four standard 3-opt reconnections: (A,B,C,D), (A,rev(B),C,D), (A,B,rev(C),D), (A,rev(B),rev(C),D). But (A,B,C,D) is original, so skip.
                    # To avoid redundancy, we only try the three non-original.
                    for cand in candidates:
                        cand_dist = compute_total_distance(cand)
                        if cand_dist < best_dist - 1e-12:
                            best_dist = cand_dist
                            best_tour = cand
                            improved = True
                            report_best_tour(best_tour)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        iter_count += 1

    return best_tour