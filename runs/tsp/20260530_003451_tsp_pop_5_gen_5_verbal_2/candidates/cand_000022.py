import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    # nearest neighbor construction
    tour = [0]
    unvisited = set(range(1, n))
    curr = 0
    while unvisited:
        nxt = min(unvisited, key=lambda c: distance_matrix[curr, c])
        tour.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    tour = np.array(tour, dtype=np.int32)

    def tour_cost(t):
        cost = 0.0
        for k in range(n):
            cost += distance_matrix[t[k], t[(k+1)%n]]
        return cost

    best_tour = tour.copy()
    best_cost = tour_cost(best_tour)
    report_best_tour(best_tour.copy())

    def best_2opt(t):
        improved = True
        while improved:
            improved = False
            best_delta = 0.0
            best_i = best_j = -1
            for i in range(n):
                for j in range(i+2, n):
                    a, b = t[i], t[(i+1)%n]
                    c, d = t[j], t[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - (distance_matrix[a,b] + distance_matrix[c,d])
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            if best_delta < 0:
                i, j = best_i, best_j
                t = np.concatenate([t[:i+1], t[j:i:-1], t[j+1:]])
                cost = tour_cost(t)
                if cost < best_cost:
                    best_cost = cost
                    best_tour = t.copy()
                    report_best_tour(best_tour.copy())
                improved = True
        return t, best_cost, best_tour

    def double_bridge(t):
        idx = sorted(np.random.choice(range(1, n-1), size=3, replace=False))
        a, b, c = idx[0], idx[1], idx[2]
        # segments: [0:a], [a:b], [b:c], [c:]
        seg1 = t[:a]
        seg2 = t[a:b]
        seg3 = t[b:c]
        seg4 = t[c:]
        # reorder: seg1, seg3, seg2, seg4
        new_t = np.concatenate([seg1, seg3, seg2, seg4])
        return new_t

    def ruin_recreate(t):
        # remove random subset of nodes (30%) and reinsert greedily
        num_remove = max(1, int(0.3 * n))
        remove_indices = np.random.choice(range(n), size=num_remove, replace=False)
        removed_nodes = t[remove_indices]
        remaining_indices = np.setdiff1d(range(n), remove_indices)
        partial_tour = t[remaining_indices]
        # reinsert removed nodes in random order greedily
        order = np.random.permutation(removed_nodes)
        for node in order:
            best_pos = -1
            best_inc = np.inf
            for pos in range(len(partial_tour)):
                prev = partial_tour[pos-1]
                nxt = partial_tour[pos]
                inc = distance_matrix[prev, node] + distance_matrix[node, nxt] - distance_matrix[prev, nxt]
                if inc < best_inc:
                    best_inc = inc
                    best_pos = pos
            partial_tour = np.insert(partial_tour, best_pos, node)
        return partial_tour

    # main loop
    max_restarts = 10
    for restart in range(max_restarts):
        t, cost, best_tour = best_2opt(tour)
        tour = t.copy()
        if cost == best_cost:
            # no improvement: perturb
            if restart < max_restarts - 1:
                # apply double-bridge
                tour = double_bridge(tour)
            else:
                # last resort: ruin-recreate
                tour = ruin_recreate(tour)
    return best_tour