import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour)
        return tour

    np.random.seed(seed)
    start = np.random.randint(n)
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        next_city = min(unvisited, key=lambda x: distance_matrix[current, x])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    tour = np.array(tour, dtype=np.int32)
    report_best_tour(tour)

    # candidate lists: nearest neighbors
    if n >= 80:
        cand_size = min(15, n-1)
    else:
        cand_size = n-1
    # For each node, store sorted indices of nearest neighbors (excluding self)
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        # argsort gives all, but we only need top cand_size
        # use argpartition for efficiency
        if cand_size < n-1:
            idx = np.argpartition(dists, cand_size)[:cand_size]
            # sort within this subset to get nearest first (optional)
            idx_sorted = idx[np.argsort(dists[idx])]
        else:
            idx_sorted = np.argsort(dists)[1:]  # skip self
        candidates.append(idx_sorted)

    # Helper to compute tour length (optional but used for delta)
    def tour_length(tour):
        total = 0.0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i+1)%n]]
        return total

    best_tour = tour.copy()
    improved = True
    eps = 1e-12
    # Main search loop
    while budget > 0 and improved:
        improved = False
        # 2-opt moves
        for i in range(n):
            if budget <= 0:
                break
            ip1 = (i+1) % n
            a = tour[i]
            b = tour[ip1]
            # iterate over candidates of b
            for j_cand in candidates[b]:
                if budget <= 0:
                    break
                # j_cand is node index; find its position in tour
                j = -1
                for idx in range(n):
                    if tour[idx] == j_cand:
                        j = idx
                        break
                if j == -1 or j == i or j == ip1 or j == (i-1) % n:
                    continue
                jp1 = (j+1) % n
                c = tour[j]
                d = tour[jp1]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                budget -= 1
                if delta < -eps:
                    # apply 2-opt: reverse segment from ip1 to j
                    if ip1 <= j:
                        tour[ip1:j+1] = tour[ip1:j+1][::-1]
                    else:
                        # wrap around case
                        segment = np.concatenate([tour[ip1:], tour[:j+1]])
                        segment = segment[::-1]
                        tour[ip1:] = segment[:n-ip1]
                        tour[:j+1] = segment[n-ip1:]
                    improved = True
                    # report if better
                    # we can compute length only on strict improvement, but delta ensures, so report
                    # to save time, skip full length calc
                    report_best_tour(tour.copy())
                    break  # first improvement exit
        # relocation moves
        for i in range(n):
            if budget <= 0:
                break
            ip1 = (i+1) % n
            v = tour[ip1]
            a = tour[i]
            # iterate over candidates of v as potential insertion point after j
            for j_cand in candidates[v]:
                if budget <= 0:
                    break
                # find position j such that tour[j] == j_cand
                j = -1
                for idx in range(n):
                    if tour[idx] == j_cand:
                        j = idx
                        break
                if j == -1 or j == i or j == ip1:
                    continue
                jp1 = (j+1) % n
                c = tour[j]
                d = tour[jp1]
                # current edges: (a,v) and (c,d); new edges: (a, next after v) and (c,v) and (v,d)
                # but careful: if jp1 == ip1? that would be same edge? avoid.
                next_v = tour[(ip1+1) % n]
                delta = distance_matrix[a, next_v] + distance_matrix[c, v] + distance_matrix[v, d] - distance_matrix[a, v] - distance_matrix[c, d] - distance_matrix[v, next_v]
                budget -= 1
                if delta < -eps:
                    # remove v from ip1, insert after j
                    # remove v
                    tour = np.delete(tour, ip1)
                    # insert after j (note j index might shift if ip1 < j? we use original indices)
                    # recompute j position after removal
                    # simpler: rebuild list
                    tlist = tour.tolist()
                    # v has been removed, find its old index not needed
                    # insert after j
                    # actually after removal, the element at position ip1 is gone
                    # But j remains same if j < ip1? messy
                    # Better: reconstruct with slice
                    # Use numpy: split tour into before v, after v, then insert after j
                    # Since v is at ip1, tour = [tour[0:ip1], tour[ip1+1:]]
                    # Then we want to insert v after position j (in original indices, but after removal, positions shift)
                    # Let's compute new position for insertion: if j < ip1: insert at j+1; if j >= ip1: insert at j (since v removed)
                    # But easier: use list manipulation
                    tlist = tour.tolist()
                    v_node = v
                    # remove v (but we already did np.delete? Actually we did tour = np.delete(tour, ip1) above? Let's keep simple: use list version
                    tlist = list(tour)
                    # find index of v in tlist (should be at original ip1, but after delete it's different)
                    # Simpler: reconstruct from original tour indices
                    # Reset: use original tour array "tour" but we are modifying in place; better to work with list conversion once
                    tlist = [int(x) for x in tour]
                    idx_v = ip1
                    v_val = tlist[idx_v]
                    del tlist[idx_v]
                    # now find insert position after j (j is index in original tour, but after deletion, if j > idx_v, insert at j; else at j+1)
                    if j > idx_v:
                        insert_pos = j  # because j shifted left by 1
                    else:
                        insert_pos = j + 1
                    # but j is index in original tour of the candidate node; after removal, candidate node's index is still j if j < idx_v, else j-1
                    # So we need to find the index of that node in the new list
                    cand_node = c  # from original tour at j
                    # find its current index
                    curr_j = tlist.index(cand_node)
                    # we want to insert after it
                    insert_pos = curr_j + 1
                    tlist.insert(insert_pos, v_val)
                    tour = np.array(tlist, dtype=np.int32)
                    improved = True
                    report_best_tour(tour.copy())
                    break
        # If no improvement, break
    return tour