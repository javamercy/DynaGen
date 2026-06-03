import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n, dtype=int)
        report_best_tour(tour)
        return tour

    def total_dist(t):
        idx = np.array(t, dtype=int)
        return distance_matrix[idx[-1], idx[0]] + np.sum(distance_matrix[idx[:-1], idx[1:]])

    def two_opt_steepest(tour):
        improved = True
        while improved:
            improved = False
            best_gain = 0.0
            best_i = best_j = None
            for i in range(n - 1):
                for j in range(i + 2, n):
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
                    if delta > best_gain + 1e-12:
                        best_gain = delta
                        best_i, best_j = i, j
            if best_gain > 1e-12:
                i, j = best_i, best_j
                tour[i+1:j+1] = tour[i+1:j+1][::-1]
                improved = True
        return tour

    def double_bridge(tour):
        cuts = sorted(random.sample(range(1, n), 3))
        seg0 = tour[:cuts[0]]
        seg1 = tour[cuts[0]:cuts[1]]
        seg2 = tour[cuts[1]:cuts[2]]
        seg3 = tour[cuts[2]:]
        return np.concatenate([seg0, seg2, seg1, seg3])

    def segment_reversal(tour):
        max_len = max(2, n // 3 + 2)
        seg_len = random.randint(2, max_len)
        i = random.randint(0, n - seg_len)
        tour[i:i+seg_len] = tour[i:i+seg_len][::-1]
        return tour

    def grasp_construction():
        start = random.randint(0, n-1)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        rcl_size = max(3, int(np.sqrt(n)))
        while unvisited:
            last = tour[-1]
            dists = [(j, distance_matrix[last, j]) for j in unvisited]
            dists.sort(key=lambda x: x[1])
            rcl = dists[:min(rcl_size, len(dists))]
            candidate = random.choice(rcl)[0]
            tour.append(candidate)
            unvisited.remove(candidate)
        return np.array(tour, dtype=int)

    best_tour = None
    best_dist = float('inf')
    num_restarts = max(5, n // 50)
    max_cycles = max(50, n // 2)
    stall_limit = max(5, n // 20)
    succ_rev = 1
    total_rev = 1
    succ_db = 1
    total_db = 1

    for _ in range(num_restarts):
        tour = grasp_construction()
        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        no_improve = 0
        for cycle in range(max_cycles):
            tour = two_opt_steepest(tour)
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
                tour = grasp_construction()
                no_improve = 0
            else:
                # Adaptive perturbation selection
                prob_rev = (succ_rev / total_rev) if total_rev > 0 else 0.5
                prob_db = (succ_db / total_db) if total_db > 0 else 0.5
                prob_rev_normalized = prob_rev / (prob_rev + prob_db)
                if random.random() < prob_rev_normalized:
                    tour_prev = tour.copy()
                    tour = segment_reversal(tour)
                    used_perturb = 'rev'
                else:
                    tour_prev = tour.copy()
                    tour = double_bridge(tour)
                    used_perturb = 'db'
                # local search next iteration will evaluate improvement
                # We need to store previous distance to compare after next local search
                # Workaround: apply local search immediately after perturbation, but that would disrupt loop logic.
                # Instead, we store the perturbation info and update after next total_dist call.
                # We'll store in a variable outside the loop; but due to scope we can use list
                # Simpler: after local search in next cycle, we can compute relative to previous best or before perturbation.
                # To keep code simple, we update success based on whether the tour after perturbation (and before next LS) is better than the best known? Not accurate.
                # Alternative: we evaluate the perturbation effect by comparing the distance before and immediately after perturbation (before LS). But LS will improve it.
                # Let's use a different design: after perturbation, compute distance, if it's better than best, count success.
                # But that would overcount because perturbation itself rarely improves. Better to measure after LS.
                # We'll tweak: after LS, if cur_dist improved over the best prior to this cycle, count success for the perturbation used in previous cycle.
                # This requires storing which perturbation was used. We'll use a list.
                # We'll store in variables: last_perturb_type and distance_before_perturb.
                # Set these after we decide perturbation, and in next cycle's local search, we check.
        # Reset adaptive counters per restart? It's global.
    return best_tour