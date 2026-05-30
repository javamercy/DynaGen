import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def total_distance(tour):
        d = 0.0
        for i in range(n-1):
            d += distance_matrix[tour[i], tour[i+1]]
        d += distance_matrix[tour[-1], tour[0]]
        return d

    # Regret insertion
    start = 0
    farthest = np.argmax(distance_matrix[start])
    tour = [start, farthest]
    unvisited = set(range(n)) - {start, farthest}
    while unvisited:
        best_cost = {}
        second_best = {}
        for city in unvisited:
            costs = []
            for i in range(len(tour)):
                prev = tour[i]
                nxt = tour[(i+1) % len(tour)]
                cost = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
                costs.append(cost)
            costs.sort()
            best_cost[city] = costs[0]
            second_best[city] = costs[1] if len(costs) > 1 else costs[0]
        regret = {c: second_best[c] - best_cost[c] for c in unvisited}
        max_regret = max(regret.values())
        candidates = [c for c in unvisited if regret[c] == max_regret]
        chosen = min(candidates, key=lambda c: best_cost[c])
        best_inc = float('inf')
        best_pos = -1
        for i in range(len(tour)):
            prev = tour[i]
            nxt = tour[(i+1) % len(tour)]
            inc = distance_matrix[prev, chosen] + distance_matrix[chosen, nxt] - distance_matrix[prev, nxt]
            if inc < best_inc:
                best_inc = inc
                best_pos = i + 1
        tour.insert(best_pos, chosen)
        unvisited.remove(chosen)
    tour = np.array(tour, dtype=int)
    best_dist = total_distance(tour)
    best_tour = tour.copy()
    report_best_tour(best_tour)

    # 2-opt local search
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+2, n):
                new_tour = np.concatenate([tour[:i+1], tour[j:i:-1], tour[j+1:]])
                new_dist = total_distance(new_tour)
                if new_dist < best_dist - 1e-12:
                    best_dist = new_dist
                    best_tour = new_tour.copy()
                    tour = best_tour
                    improved = True
                    report_best_tour(best_tour)
                    break
            if improved:
                break

    # Double-bridge perturbation
    a, b, c = sorted(np.random.choice(range(1, n), size=3, replace=False))
    seg1 = tour[:a]
    seg2 = tour[a:b]
    seg3 = tour[b:c]
    seg4 = tour[c:]
    new_tour = np.concatenate([seg1, seg3, seg2, seg4])
    new_dist = total_distance(new_tour)
    if new_dist < best_dist - 1e-12:
        best_dist = new_dist
        best_tour = new_tour.copy()
        report_best_tour(best_tour)

    # Refine perturbed tour with 2-opt
    tour = best_tour.copy()
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+2, n):
                new_tour = np.concatenate([tour[:i+1], tour[j:i:-1], tour[j+1:]])
                new_dist = total_distance(new_tour)
                if new_dist < best_dist - 1e-12:
                    best_dist = new_dist
                    best_tour = new_tour.copy()
                    tour = best_tour
                    improved = True
                    report_best_tour(best_tour)
                    break
            if improved:
                break

    return best_tour