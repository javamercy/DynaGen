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

    # Regret insertion with tie-breaking (smallest best cost)
    def construct():
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
        return np.array(tour, dtype=int)

    # 2-opt improvement
    def two_opt(tour, dist):
        improved = True
        n_ = len(tour)
        while improved:
            improved = False
            for i in range(n_):
                for j in range(i+2, n_):
                    if j - i == 1:
                        continue
                    new_tour = np.concatenate([tour[:i+1], tour[j:i:-1], tour[j+1:]])
                    new_dist = total_distance(new_tour)
                    if new_dist < dist - 1e-12:
                        tour, dist = new_tour, new_dist
                        improved = True
                        # report_best_tour(tour)
        return tour, dist

    # Or-opt improvement (chain lengths 1,2,3)
    def or_opt(tour, dist):
        improved = True
        n_ = len(tour)
        while improved:
            improved = False
            for chain_len in [1, 2, 3]:
                for start in range(n_):
                    chain = []
                    for k in range(chain_len):
                        chain.append(tour[(start + k) % n_])
                    tour_list = tour.tolist()
                    for city in chain:
                        tour_list.remove(city)
                    for pos in range(len(tour_list) + 1):
                        cand = tour_list[:pos] + chain + tour_list[pos:]
                        cand = np.array(cand, dtype=int)
                        cand_dist = total_distance(cand)
                        if cand_dist < dist - 1e-12:
                            tour, dist = cand, cand_dist
                            improved = True
                            # report_best_tour(tour)
                            break
                    if improved:
                        break
                if improved:
                    break
        return tour, dist

    # Full local search
    def local_search(tour):
        dist = total_distance(tour)
        tour, dist = two_opt(tour, dist)
        tour, dist = or_opt(tour, dist)
        tour, dist = two_opt(tour, dist)
        return tour, dist

    # Double-bridge perturbation
    def double_bridge(tour):
        n_ = len(tour)
        # pick cutting points: ensure segments at least 1 city
        i = np.random.randint(0, n_//4)
        j = np.random.randint(i+2, min(n_-2, i+ n_//2))
        k = np.random.randint(j+2, min(n_-1, j+ n_//2))
        l = np.random.randint(k+2, n_)
        # if any segment empty, adjust? but with n>=4, this should be okay.
        # reassemble: [i:j] becomes [k:l] then [j:k] then [i:j] then [l:]
        part1 = tour[:i]
        part2 = tour[i:j]
        part3 = tour[j:k]
        part4 = tour[k:l]
        part5 = tour[l:]
        new_tour = np.concatenate([part1, part4, part3, part2, part5])
        return new_tour

    # Construction
    best_tour = construct()
    best_tour, best_dist = local_search(best_tour)
    # report_best_tour(best_tour)

    # Iterated local search with perturbations
    for _ in range(5):
        perturbed = double_bridge(best_tour)
        perturbed, _ = local_search(perturbed)
        if total_distance(perturbed) < best_dist - 1e-12:
            best_tour = perturbed
            best_dist = total_distance(best_tour)
            # report_best_tour(best_tour)
        else:
            # if no improvement, maybe revert, but keep best
            pass

    return best_tour