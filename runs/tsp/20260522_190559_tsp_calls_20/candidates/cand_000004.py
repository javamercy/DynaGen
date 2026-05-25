import numpy as np
import random
import math

def report_best_tour(tour):
    # Placeholder for reporting mechanism
    pass

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 1:
        return np.arange(n)
    if n == 2:
        return np.array([0, 1])

    random.seed(seed)
    np.random.seed(seed)

    # 1. Initial Tour: Random Shuffle
    tour = np.arange(n)
    # Fix 0 as start, shuffle others
    perm = list(range(1, n))
    random.shuffle(perm)
    tour = np.concatenate(([0], np.array(perm)))
    
    def get_tour_dist(t):
        d = 0
        for i in range(n):
            d += distance_matrix[t[i], t[(i + 1) % n]]
        return d

    current_dist = get_tour_dist(tour)
    best_tour = tour.copy()
    best_dist = current_dist
    report_best_tour(best_tour)

    # 2. Simulated Annealing Parameters
    # T_start and T_end are heuristics; budget controls iterations
    t_start = 100.0
    t_end = 0.01
    temp = t_start
    cooling_rate = math.exp(math.log(t_end / t_start) / budget) if budget > 0 else 1.0

    # 3. Search Loop
    for i in range(budget):
        # Select two random indices for 2-opt swap
        idx1 = random.randint(0, n - 1)
        idx2 = random.randint(0, n - 1)
        if idx1 == idx2:
            continue
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        # Calculate delta for 2-opt swap
        # Edges involved: (idx1, idx1+1) and (idx2, idx2+1)
        u = tour[idx1]
        v = tour[(idx1 + 1) % n]
        x = tour[idx2]
        y = tour[(idx2 + 1) % n]
        
        # Note: This is a simplification of the 2-opt delta
        # Standard 2-opt reverses the segment between v and x
        # New edges: (u, x) and (v, y)
        delta = (distance_matrix[u, x] + distance_matrix[v, y]) - \
                (distance_matrix[u, v] + distance_matrix[x, y])

        # Acceptance criteria (Metropolis)
        if delta < 0 or random.random() < math.exp(-delta / temp):
            # Perform reversal
            new_tour = tour.copy()
            # Reverse segment from idx1+1 to idx2
            start, end = idx1 + 1, idx2
            # Handle wrap-around if necessary, but simple slice is safer for indices
            if start <= end:
                new_tour[start:end+1] = tour[end:start-1 if start > 0 else -1:-1]
                # Correcting slice logic for the wrap-around case is complex,
                # so we use a simple list reversal for robustness
                t_list = tour.tolist()
                l, r = start, end
                while l < r:
                    t_list[l], t_list[r] = t_list[r], t_list[l]
                    l += 1
                    r -= 1
                new_tour = np.array(t_list)
            
            current_dist += delta
            tour = new_tour
            
            if current_dist < best_dist:
                best_dist = current_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
        
        temp *= cooling_rate

    return best_tour