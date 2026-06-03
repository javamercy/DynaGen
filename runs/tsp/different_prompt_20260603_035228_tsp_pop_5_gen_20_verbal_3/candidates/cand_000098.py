import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n)

    def total_dist(tour):
        return distance_matrix[tour[-1], tour[0]] + np.sum(distance_matrix[tour[:-1], tour[1:]])

    def two_opt_first(tour, dist):
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == n - 1:
                        delta = (dist[tour[i], tour[i+1]] +
                                 dist[tour[j], tour[0]] -
                                 dist[tour[i], tour[j]] -
                                 dist[tour[i+1], tour[0]])
                    else:
                        delta = (dist[tour[i], tour[i+1]] +
                                 dist[tour[j], tour[j+1]] -
                                 dist[tour[i], tour[j]] -
                                 dist[tour[i+1], tour[j+1]])
                    if delta < -1e-12:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return tour

    def double_bridge(tour):
        cuts = sorted(np.random.choice(range(1, n), 3, replace=False))
        seg0 = tour[:cuts[0]]
        seg1 = tour[cuts[0]:cuts[1]]
        seg2 = tour[cuts[1]:cuts[2]]
        seg3 = tour[cuts[2]:]
        return np.concatenate([seg0, seg2, seg1, seg3])

    best_tour = None
    best_dist = float('inf')
    num_restarts = 8
    max_cycles = 15
    stall_limit = 4

    for _ in range(num_restarts):
        # Randomized nearest neighbor with top-3
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            last = tour[-1]
            unvisited_list = list(unvisited)
            dists = distance_matrix[last, unvisited_list]
            # top k = min(3, len(unvisited))
            k = min(3, len(unvisited))
            idx_sorted = np.argsort(dists)
            top_idx = idx_sorted[:k]
            top_dists = dists[top_idx]
            inv = 1.0 / (top_dists + 1e-12)
            probs = inv / inv.sum()
            choice = np.random.choice(top_idx, p=probs)
            next_city = unvisited_list[choice]
            tour.append(next_city)
            unvisited.remove(next_city)
        tour = np.array(tour, dtype=int)
        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)

        no_improve = 0
        for cycle in range(max_cycles):
            tour = two_opt_first(tour, distance_matrix)
            cur_dist = total_dist(tour)
            if cur_dist < best_dist - 1e-12:
                best_dist = cur_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
                no_improve = 0
            else:
                no_improve += 1

            if cycle == max_cycles - 1:
                break

            if no_improve >= stall_limit:
                tour = double_bridge(tour)
                cur_dist = total_dist(tour)
                no_improve = 0
            else:
                # Small perturbation: reverse random segment
                seg_len = np.random.randint(2, max(3, n // 10 + 2))
                start_idx = np.random.randint(0, n - seg_len + 1)
                tour[start_idx:start_idx+seg_len] = tour[start_idx:start_idx+seg_len][::-1]
                cur_dist = total_dist(tour)

    return best_tour