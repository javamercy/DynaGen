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
    # candidate list (nearest neighbors)
    cand_size = max(10, min(40, n // 2))
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        order = np.argsort(dists)
        order = order[order != i][:cand_size]
        candidates.append(set(order))
    
    # helper to compute tour distance
    def tour_distance(t):
        d = 0.0
        for idx in range(n):
            d += distance_matrix[t[idx], t[(idx+1)%n]]
        return d
    
    # helper to compute position array
    def get_pos(t):
        pos = np.empty(n, dtype=int)
        for idx, node in enumerate(t):
            pos[node] = idx
        return pos
    
    # initial tour: random nearest neighbor
    start = rng.randrange(n)
    tour = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    cur = start
    while unvisited:
        # choose among nearest neighbors with roulette
        neighbors = []
        for v in unvisited:
            d = distance_matrix[cur, v]
            neighbors.append((v, d))
        # sort by distance, take top k
        neighbors.sort(key=lambda x: x[1])
        k = min(cand_size, len(neighbors))
        top = neighbors[:k]
        # roulette: weight = 1/(d+1e-12)
        weights = [1.0/(d+1e-12) for v,d in top]
        total = sum(weights)
        r = rng.random() * total
        cum = 0.0
        chosen = top[-1][0]
        for (v,d), w in zip(top, weights):
            cum += w
            if r <= cum:
                chosen = v
                break
        tour.append(chosen)
        unvisited.remove(chosen)
        cur = chosen
    tour = np.array(tour, dtype=int)
    current_dist = tour_distance(tour)
    best_tour = tour.copy()
    best_dist = current_dist
    try:
        report_best_tour(best_tour)
    except:
        pass
    pos = get_pos(tour)
    total_evals = 0
    improved = True
    # main loop
    while total_evals < budget:
        # 2-opt phase
        improved_local = True
        while improved_local and total_evals < budget:
            improved_local = False
            # build list of candidate (i,j) pairs
            pairs = []
            for i in range(n):
                a = tour[i]
                for b in candidates[a]:
                    j = pos[b]
                    if j <= i:
                        continue
                    if (i+1)%n == j or (j+1)%n == i:
                        continue
                    pairs.append((i, j))
            rng.shuffle(pairs)
            for i, j in pairs:
                if total_evals >= budget:
                    break
                a = tour[i]
                b = tour[(i+1)%n]
                c = tour[j]
                d = tour[(j+1)%n]
                old = distance_matrix[a, b] + distance_matrix[c, d]
                new = distance_matrix[a, c] + distance_matrix[b, d]
                delta = new - old
                total_evals += 1
                if delta < -1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    pos = get_pos(tour)
                    current_dist += delta
                    if current_dist < best_dist - 1e-12:
                        best_dist = current_dist
                        best_tour = tour.copy()
                        try:
                            report_best_tour(best_tour)
                        except:
                            pass
                    improved_local = True
                    break
        if not improved_local and total_evals < budget:
            # double-bridge perturbation
            # pick 4 distinct indices, not necessarily evenly spaced
            # ensure each segment has at least 1 node
            while True:
                idx = sorted(rng.sample(range(n), 4))
                # segments: [0..a-1], [a..b-1], [b..c-1], [c..d-1], [d..n-1] with wrap
                # but we treat tour as cyclic, so just break at indices
                a, b, c, d = idx
                if a > 0 and b > a+1 and c > b+1 and d > c+1:
                    break
            # apply double bridge: reorder segments
            # segments: [0:a], [a:b], [b:c], [c:d], [d:n]
            # new order: [0:a], [c:d], [b:c], [a:b], [d:n]
            # but careful with cyclic: if a==0, then segment [0:a] is empty
            seg1 = tour[:a]
            seg2 = tour[a:b]
            seg3 = tour[b:c]
            seg4 = tour[c:d]
            seg5 = tour[d:]
            new_tour = np.concatenate([seg1, seg4, seg3, seg2, seg5])
            # ensure valid tour: may duplicate? No, segments are disjoint
            tour = new_tour
            pos = get_pos(tour)
            current_dist = tour_distance(tour)
            total_evals += 1  # perturbation counts as evaluation
            # update best if better
            if current_dist < best_dist - 1e-12:
                best_dist = current_dist
                best_tour = tour.copy()
                try:
                    report_best_tour(best_tour)
                except:
                    pass
            # reset improvement flag
            improved_local = True
    return best_tour