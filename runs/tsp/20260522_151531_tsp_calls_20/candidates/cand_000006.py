import numpy as np
import random

def report_best_tour(tour):
    pass

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        res = np.arange(n)
        report_best_tour(res)
        return res

    random.seed(seed)
    np.random.seed(seed)

    def get_dist(tour):
        d = 0
        for i in range(n):
            d += distance_matrix[tour[i], tour[(i + 1) % n]]
        return d

    # 1. Construction: Greedy Nearest Neighbor (from cand_000001)
    unvisited = set(range(1, n))
    tour_list = [0]
    curr = 0
    while unvisited:
        next_node = min(unvisited, key=lambda x: distance_matrix[curr, x])
        tour_list.append(next_node)
        unvisited.remove(next_node)
        curr = next_node
    
    current_tour = np.array(tour_list)
    best_tour = current_tour.copy()
    best_dist = get_dist(best_tour)
    report_best_tour(best_tour)

    # 2. Candidate List for scaling (from cand_000001)
    k_neighbors = 20 if n >= 80 else n - 1
    candidates = []
    for i in range(n):
        idx = np.argsort(distance_matrix[i])[:k_neighbors + 1]
        candidates.append(set(idx))

    def local_search(tour, current_dist, budget_limit):
        nonlocal best_dist, best_tour
        improved = True
        iters = 0
        while improved and iters < budget_limit:
            improved = False
            # Use a limited scan to avoid O(n^2) in every single inner loop
            for i in range(n - 1):
                # Only check a few j's or use candidates to keep it fast
                # For simplicity and robustness, we check a window or random samples
                for j in range(i + 2, n):
                    u, v = tour[i], tour[i+1]
                    w, z = tour[j], tour[(j+1)%n]
                    
                    # Candidate check: only swap if endpoints are reasonably close
                    if n < 80 or (u in candidates[w] or v in candidates[z]):
                        delta = (distance_matrix[u, w] + distance_matrix[v, z]) - \
                                (distance_matrix[u, v] + distance_matrix[w, z])
                        
                        if delta < -1e-9:
                            tour[i+1:j+1] = tour[i+1:j+1][::-1]
                            current_dist += delta
                            improved = True
                            if current_dist < best_dist:
                                best_dist = current_dist
                                best_tour = tour.copy()
                                report_best_tour(best_tour)
                            break
                if improved: break
            iters += 1
        return tour, current_dist

    def perturb(tour):
        # Double-bridge move (from cand_000003)
        new_tour = tour.copy()
        indices = sorted(random.sample(range(n), 4))
        i, j, k, l = indices
        seg1 = new_tour[:i+1]
        seg2 = new_tour[i+1:j+1]
        seg3 = new_tour[j+1:k+1]
        seg4 = new_tour[k+1:]
        return np.concatenate([seg1, seg3, seg2, seg4])

    # 3. ILS Loop
    total_iters = 0
    # Divide budget between local search and perturbations
    # Since 2-opt is expensive, we limit the number of full local search passes
    while total_iters < budget:
        # Intensive local search phase
        current_tour, current_dist = local_search(current_tour, best_dist, budget // 10 + 1)
        
        # Diversification phase
        current_tour = perturb(current_tour)
        
        total_iters += 1
        # Safety break if we've spent too much time (budget as a proxy for iterations)
        if total_iters > budget: break

    return best_tour