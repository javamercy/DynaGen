import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n, dtype=int)
        report_best_tour(tour)
        return tour

    def total_dist(t):
        idx = np.array(t, dtype=int)
        return distance_matrix[idx[-1], idx[0]] + np.sum(distance_matrix[idx[:-1], idx[1:]])

    def farthest_insertion_construction():
        remaining = set(range(n))
        start = np.random.randint(n)
        tour = [start]
        remaining.remove(start)
        # second node: nearest to start
        nearest = min(remaining, key=lambda x: distance_matrix[start, x])
        tour.append(nearest)
        remaining.remove(nearest)
        while remaining:
            # find node with max distance to tour (minimum distance to any tour node)
            farthest_node = None
            max_min_dist = -1.0
            for node in remaining:
                min_dist = min(distance_matrix[node, t] for t in tour)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    farthest_node = node
            # best insertion position that minimizes increase
            best_pos = None
            best_increase = float('inf')
            for pos in range(len(tour)):
                # insert between pos and pos+1 (wrap)
                a = tour[pos]
                b = tour[(pos + 1) % len(tour)]
                increase = distance_matrix[a, farthest_node] + distance_matrix[farthest_node, b] - distance_matrix[a, b]
                if increase < best_increase:
                    best_increase = increase
                    best_pos = pos + 1  # insert after pos
            tour.insert(best_pos, farthest_node)
            remaining.remove(farthest_node)
        return np.array(tour, dtype=int)

    def two_opt_first(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    # delta for reversing segment [i+1:j+1]
                    if j == n - 1:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[0]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[0]])
                    else:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[j+1]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[j+1]])
                    if delta > 1e-12:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
                        break  # restart scanning after first improvement
                if improved:
                    break
        return tour

    best_tour = None
    best_dist = float('inf')
    num_restarts = 3
    max_cycles = 20
    stall_limit = 5

    for _ in range(num_restarts):
        tour = farthest_insertion_construction()
        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)

        no_improve = 0
        for cycle in range(max_cycles):
            tour = two_opt_first(tour)
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

            # perturbation
            if no_improve >= stall_limit:
                # random reversal of a segment of length at least 2
                seg_len = np.random.randint(2, max(3, n//2 + 1))
                i = np.random.randint(0, n - seg_len)
                tour[i:i+seg_len] = tour[i:i+seg_len][::-1]
                no_improve = 0
            else:
                # small perturbation if no_improve == 0 ? Actually we always perturb after a cycle but not if stalled? 
                # To avoid too much perturbation, only apply if no_improve == 0? 
                # We'll just do a small reversal every cycle if not stalled (as per typical ILS)
                if no_improve < stall_limit:
                    seg_len = np.random.randint(2, max(3, n//4 + 1))
                    i = np.random.randint(0, n - seg_len)
                    tour[i:i+seg_len] = tour[i:i+seg_len][::-1]
                else:
                    # already handled above
                    pass
        # end cycle loop
    return best_tour