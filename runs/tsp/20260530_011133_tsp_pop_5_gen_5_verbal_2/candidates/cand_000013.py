import numpy as np
from copy import deepcopy

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        return np.array([0])
    # farthest insertion construction
    start, end = np.unravel_index(np.argmax(distance_matrix), (n, n))
    tour = [start, end]
    unvisited = set(range(n)) - {start, end}
    while unvisited:
        farthest_city = max(unvisited, key=lambda c: min(distance_matrix[c][t] for t in tour))
        best_pos = min(range(1, len(tour)+1), key=lambda i: distance_matrix[tour[i-1]][farthest_city] + distance_matrix[farthest_city][tour[i%len(tour)]] - distance_matrix[tour[i-1]][tour[i%len(tour)]])
        tour.insert(best_pos, farthest_city)
        unvisited.remove(farthest_city)
    best_tour = np.array(tour)
    report_best_tour(best_tour)
    
    def tour_len(t):
        return distance_matrix[t[-1], t[0]] + sum(distance_matrix[t[i], t[i+1]] for i in range(n-1))
    
    def two_opt(t):
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for k in range(i+2, n):
                    if i == 0 and k == n-1:
                        continue
                    new_t = t.copy()
                    new_t[i+1:k+1] = t[i+1:k+1][::-1]
                    delta = (distance_matrix[t[i], new_t[i+1]] + distance_matrix[new_t[k], t[(k+1)%n]] -
                             distance_matrix[t[i], t[i+1]] - distance_matrix[t[k], t[(k+1)%n]])
                    if delta < -1e-10:
                        t = new_t
                        improved = True
        return t
    
    def or_opt(t):
        improved = True
        while improved:
            improved = False
            for L in [1, 2, 3]:
                for i in range(n):
                    j = i + L
                    if j > n:
                        break
                    segment = t[i:j].tolist()
                    # remove segment
                    rest = t[:i].tolist() + t[j:].tolist()
                    if len(rest) == 0:
                        continue
                    # best insertion position in rest
                    best_pos = 0
                    best_inc = float('inf')
                    for pos in range(len(rest)+1):
                        # insert segment as block
                        new_rest = rest[:pos] + segment + rest[pos:]
                        # compute length
                        Lnew = 0
                        for idx in range(n-1):
                            Lnew += distance_matrix[new_rest[idx], new_rest[idx+1]]
                        Lnew += distance_matrix[new_rest[-1], new_rest[0]]
                        if Lnew < best_inc:
                            best_inc = Lnew
                            best_pos = pos
                    new_t = np.array(rest[:best_pos] + segment + rest[best_pos:])
                    if best_inc < tour_len(t) - 1e-10:
                        t = new_t
                        improved = True
        return t
    
    def double_bridge(t):
        # random cut points
        a = np.random.randint(1, n//2)
        b = np.random.randint(a+1, n-2)
        c = np.random.randint(b+1, n-1)
        d = n
        # segments: [0:a], [a:b], [b:c], [c:d]
        # reorder: 1st, 3rd, 2nd, 4th
        new_t = np.concatenate([t[:a], t[c:d], t[b:c], t[a:b]])
        return new_t
    
    # Iterated local search
    current = best_tour.copy()
    for _ in range(20):
        current = two_opt(current)
        current = or_opt(current)
        if tour_len(current) < tour_len(best_tour) - 1e-10:
            best_tour = current.copy()
            report_best_tour(best_tour)
        # perturbation
        current = double_bridge(current)
        # after perturbation, 2-opt and Or-opt again
        current = two_opt(current)
        current = or_opt(current)
        if tour_len(current) < tour_len(best_tour) - 1e-10:
            best_tour = current.copy()
            report_best_tour(best_tour)
    return best_tour