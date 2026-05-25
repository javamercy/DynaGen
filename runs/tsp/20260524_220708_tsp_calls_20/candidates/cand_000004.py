import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(distance_matrix)
    # nearest neighbor construction from node 0
    tour = [0]
    visited = {0}
    cur = 0
    for _ in range(n - 1):
        next = min((j for j in range(n) if j not in visited), key=lambda j: distance_matrix[cur, j])
        tour.append(next)
        visited.add(next)
        cur = next
    tour = np.array(tour, dtype=np.int64)
    best_tour = tour.copy()
    best_cost = distance_matrix[tour[-1], tour[0]] + sum(distance_matrix[tour[i], tour[i+1]] for i in range(n-1))
    report_best_tour(best_tour)
    
    def compute_cost(t):
        return distance_matrix[t[-1], t[0]] + sum(distance_matrix[t[i], t[i+1]] for i in range(n-1))
    
    cost = best_cost
    iter_count = 0
    improved = True
    while iter_count < budget and improved:
        improved = False
        # generate random pairs for 2-opt
        pairs = [(i, j) for i in range(n) for j in range(i+2, n)]
        rng.shuffle(pairs)
        for i, j in pairs:
            # compute delta
            a, b, c, d = tour[i], tour[(i+1)%n], tour[j], tour[(j+1)%n]
            delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
            if delta < -1e-9:
                # apply 2-opt reversal
                if i+1 < j:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                else:
                    # wrap-around case not handled by simple reversal; skip for simplicity
                    continue
                cost += delta
                improved = True
                iter_count += 1
                if cost < best_cost:
                    best_cost = cost
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                if iter_count >= budget:
                    break
        # if no improvement in full loop, improved remains False and while ends
    return best_tour