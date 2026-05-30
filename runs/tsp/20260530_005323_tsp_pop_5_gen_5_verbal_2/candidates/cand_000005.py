import numpy as np
import random

def solve_tsp(dm):
    n = dm.shape[0]
    best_tour = None
    best_cost = float('inf')
    for _ in range(10):
        start = random.randint(0, n-1)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        cur = start
        while unvisited:
            dists = [(dm[cur, j], j) for j in unvisited]
            dists.sort(key=lambda x: x[0])
            k = min(3, len(dists))
            cand = dists[:k]
            _, nxt = random.choice(cand)
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a, b = tour[i], tour[(i+1)%n]
                    c, d = tour[j], tour[(j+1)%n]
                    delta = dm[a, c] + dm[b, d] - dm[a, b] - dm[c, d]
                    if delta < -1e-12:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
                        break
                if improved:
                    break
        cost = sum(dm[tour[k], tour[(k+1)%n]] for k in range(n))
        if cost < best_cost:
            best_cost = cost
            best_tour = np.array(tour)
            report_best_tour(best_tour)
    return best_tour