import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n == 1:
        tour = np.array([0])
        report_best_tour(tour)
        return tour
    
    # Farthest insertion construction
    tour = [0]
    unvisited = set(range(1, n))
    while unvisited:
        best_node = None
        best_min_dist = -1
        for v in unvisited:
            min_dist = min(distance_matrix[v, t] for t in tour)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_node = v
        best_increase = np.inf
        best_pos = 0
        for i in range(len(tour)):
            prev = tour[i]
            nxt = tour[(i+1) % len(tour)]
            increase = distance_matrix[prev, best_node] + distance_matrix[best_node, nxt] - distance_matrix[prev, nxt]
            if increase < best_increase:
                best_increase = increase
                best_pos = i+1
        tour.insert(best_pos, best_node)
        unvisited.remove(best_node)
    tour = np.array(tour, dtype=np.int32)
    best_tour = tour.copy()
    best_cost = 0.0
    for i in range(n):
        best_cost += distance_matrix[tour[i], tour[(i+1)%n]]
    report_best_tour(best_tour)
    
    # 3-opt local search with random restarts
    max_restarts = 30
    for restart in range(max_restarts):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    for k in range(j+2, n):
                        # consider reversing segments between i+1..j and j+1..k in various ways
                        a, b = tour[i], tour[(i+1)%n]
                        c, d = tour[j], tour[(j+1)%n]
                        e, f = tour[k], tour[(k+1)%n]
                        # compute current cost of these three edges
                        curr = distance_matrix[a,b] + distance_matrix[c,d] + distance_matrix[e,f]
                        # try all 3-opt moves (4 possible new connections after removing 3 edges)
                        # case 1: reverse segment (i+1..j) -> new edges: a-c, b-d, e-f (but order matters)
                        # Actually, we loop over all 4 possible reconnections (excluding identity)
                        # Move type 0: a-c, b-d, e-f
                        d1 = distance_matrix[a,c] + distance_matrix[b,d] + distance_matrix[e,f]
                        if d1 < curr:
                            # reverse i+1..j
                            tour[i+1:j+1] = tour[i+1:j+1][::-1]
                            improved = True
                            break
                        # Move type 1: a-c, b-e, d-f
                        d2 = distance_matrix[a,c] + distance_matrix[b,e] + distance_matrix[d,f]
                        if d2 < curr:
                            # reverse i+1..k and then rotate? Simpler: reconstruct segments
                            # We'll just rebuild by creating new tour
                            seg1 = tour[i+1:j+1]
                            seg2 = tour[j+1:k+1]
                            new_tour = np.concatenate([tour[:i+1], seg2, seg1, tour[k+1:]])
                            tour = new_tour
                            improved = True
                            break
                        # Move type 2: a-d, b-c, e-f
                        d3 = distance_matrix[a,d] + distance_matrix[b,c] + distance_matrix[e,f]
                        if d3 < curr:
                            # reverse j+1..k and then swap? Simpler: rearrange
                            seg1 = tour[i+1:j+1]
                            seg2 = tour[j+1:k+1]
                            new_tour = np.concatenate([tour[:i+1], seg1[::-1], seg2[::-1], tour[k+1:]])
                            tour = new_tour
                            improved = True
                            break
                        # Move type 3: a-e, b-c, d-f
                        d4 = distance_matrix[a,e] + distance_matrix[b,c] + distance_matrix[d,f]
                        if d4 < curr:
                            # complex rearrangement
                            seg1 = tour[i+1:j+1]
                            seg2 = tour[j+1:k+1]
                            new_tour = np.concatenate([tour[:i+1], seg2[::-1], seg1, tour[k+1:]])
                            tour = new_tour
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            # update best if needed
            cost = 0.0
            for i in range(n):
                cost += distance_matrix[tour[i], tour[(i+1)%n]]
            if cost < best_cost:
                best_cost = cost
                best_tour = tour.copy()
                report_best_tour(best_tour)
        # after local optimum, apply double-bridge perturbation (if not last restart)
        if restart == max_restarts - 1:
            break
        # double-bridge: cut four edges and reconnect
        # pick four random indices
        idx = np.random.choice(n, 4, replace=False)
        idx.sort()
        a, b, c, d = idx
        # ensure non-adjacent
        if b - a < 2 or c - b < 2 or d - c < 2:
            continue
        # take segments
        seg1 = tour[a:b]
        seg2 = tour[b:c]
        seg3 = tour[c:d]
        seg4 = tour[d:]
        # reconnect: start, seg3, seg2, seg1, seg4
        tour = np.concatenate([tour[:a], seg3, seg2, seg1, tour[d:]])
        # ensure circular correctness? Already ok
    return best_tour