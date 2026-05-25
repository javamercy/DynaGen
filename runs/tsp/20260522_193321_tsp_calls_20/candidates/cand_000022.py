import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=int)
        try:
            report_best_tour(tour)
        except:
            pass
        return tour
    rng = random.Random(seed)
    # Candidate lists
    cand_size = 30 if n < 80 else 20
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        order = np.argsort(dists)
        order = order[order != i][:cand_size]
        candidates.append(set(order))
    
    total_evals = 0
    best_tour = None
    best_dist = float('inf')
    
    # Use a single restart (could add more if budget remains, but we keep it simple)
    max_restarts = 1
    for restart in range(max_restarts):
        # Regret-2 construction
        start = rng.randrange(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            best_city = None
            best_regret = -1.0
            best_pos = None
            for city in unvisited:
                min_cost = float('inf')
                second_min = float('inf')
                best_idx = -1
                for idx in range(len(tour)):
                    if idx == len(tour) - 1:
                        cost = distance_matrix[tour[idx], city] + distance_matrix[city, tour[0]] - distance_matrix[tour[idx], tour[0]]
                    else:
                        cost = distance_matrix[tour[idx], city] + distance_matrix[city, tour[idx+1]] - distance_matrix[tour[idx], tour[idx+1]]
                    if cost < min_cost:
                        second_min = min_cost
                        min_cost = cost
                        best_idx = idx
                    elif cost < second_min:
                        second_min = cost
                regret = second_min - min_cost
                if regret > best_regret:
                    best_regret = regret
                    best_city = city
                    best_pos = best_idx
            tour.insert(best_pos + 1, best_city)
            unvisited.remove(best_city)
        tour = np.array(tour, dtype=int)
        # Compute initial distance
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour[i], tour[(i+1)%n]]
        if dist < best_dist:
            best_dist = dist
            best_tour = tour.copy()
            try:
                report_best_tour(best_tour)
            except:
                pass
        # Position array
        pos = np.empty(n, dtype=int)
        for idx, node in enumerate(tour):
            pos[node] = idx
        # Best-improvement 2-opt
        improved = True
        while total_evals < budget and improved:
            improved = False
            # Generate all candidate moves
            pairs = []
            for i in range(n):
                a = tour[i]
                for b in candidates[a]:
                    j = pos[b]
                    if j <= i:
                        continue
                    if (i+1)%n == j or (j+1)%n == i:
                        continue
                    # compute delta without applying
                    a_city = tour[i]
                    b_city = tour[(i+1)%n]
                    c_city = tour[j]
                    d_city = tour[(j+1)%n]
                    old = distance_matrix[a_city, b_city] + distance_matrix[c_city, d_city]
                    new = distance_matrix[a_city, c_city] + distance_matrix[b_city, d_city]
                    delta = new - old
                    pairs.append((delta, i, j))
            if not pairs:
                break
            # Shuffle for tie-breaking
            rng.shuffle(pairs)
            # Find best delta
            best_delta = 0.0
            best_i = -1
            best_j = -1
            for delta, i, j in pairs:
                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_i = i
                    best_j = j
            # Count evaluations: we evaluated all pairs, but to be budget-correct, count each pair
            # However budget is for search effort; we can count each delta computation as an evaluation.
            total_evals += len(pairs)
            if total_evals > budget:
                # Exceeded, but we already computed; we should stop after this iteration
                # To be safe, cap at budget and break after applying if we have budget left?
                # Better: check if total_evals >= budget and break before applying
                if total_evals >= budget:
                    break
            if best_delta < -1e-12:
                i, j = best_i, best_j
                # Apply 2-opt
                if i < j:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                else:
                    segment = np.concatenate((tour[i+1:], tour[:j+1]))
                    segment = segment[::-1]
                    tour[i+1:] = segment[:n-i-1]
                    tour[:j+1] = segment[n-i-1:]
                # Update positions
                for idx in range(n):
                    pos[tour[idx]] = idx
                dist += best_delta
                if dist < best_dist - 1e-12:
                    best_dist = dist
                    best_tour = tour.copy()
                    try:
                        report_best_tour(best_tour)
                    except:
                        pass
                improved = True
    return best_tour