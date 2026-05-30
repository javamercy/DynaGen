import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n)

    # Farthest insertion construction
    max_dist = -1
    start = end = 0
    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i, j] > max_dist:
                max_dist = distance_matrix[i, j]
                start, end = i, j
    tour = [start, end]
    in_tour = {start, end}
    while len(tour) < n:
        farthest = None
        max_min = -1
        for city in range(n):
            if city in in_tour:
                continue
            min_dist = min(distance_matrix[city, t] for t in tour)
            if min_dist > max_min:
                max_min = min_dist
                farthest = city
        best_pos = 0
        best_inc = float('inf')
        for pos in range(len(tour)):
            prev = tour[pos]
            nxt = tour[(pos+1) % len(tour)]
            inc = distance_matrix[prev, farthest] + distance_matrix[farthest, nxt] - distance_matrix[prev, nxt]
            if inc < best_inc:
                best_inc = inc
                best_pos = pos + 1
        tour.insert(best_pos, farthest)
        in_tour.add(farthest)
    tour = np.array(tour)
    best_dist = compute_dist(tour, distance_matrix)
    report_best_tour(tour)

    # Helper to compute distance
    def compute_dist(t, dm):
        return sum(dm[t[i], t[(i+1) % n]] for i in range(n))

    # Local search loop
    improved = True
    while improved:
        improved = False
        # Best-improvement 2-opt
        best_i = best_j = -1
        best_delta = 0
        for i in range(n-2):
            for j in range(i+2, n):
                a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1) % n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        if best_delta < 0:
            # Apply 2-opt move
            tour[best_i+1:best_j+1] = tour[best_i+1:best_j+1][::-1]
            improved = True
            new_dist = best_dist + best_delta
            if new_dist < best_dist:
                best_dist = new_dist
                report_best_tour(tour)
            # Recompute best_dist to avoid floating error? We'll trust delta.
            best_dist = new_dist

        # Best-improvement node insertion (relocate single city)
        best_city = -1
        best_from_pos = -1
        best_to_pos = -1
        best_delta = 0
        for pos in range(n):
            city = tour[pos]
            # Remove city temporarily: compute distance saved
            prev = tour[(pos-1) % n]
            nxt = tour[(pos+1) % n]
            removed_saving = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
            # Try inserting at every other position
            for target in range(n):
                if target == pos or target == (pos+1) % n:
                    continue
                # Insert between tour[target-1] and tour[target]
                before = tour[(target-1) % n]
                after = tour[target % n]
                new_edges = distance_matrix[before, city] + distance_matrix[city, after] - distance_matrix[before, after]
                delta = new_edges - removed_saving
                if delta < best_delta:
                    best_delta = delta
                    best_city = city
                    best_from_pos = pos
                    best_to_pos = target
        if best_delta < 0:
            # Apply move: remove city from best_from_pos and insert at best_to_pos
            # Remove city
            tour = np.delete(tour, best_from_pos)
            # Adjust insertion index if necessary
            if best_to_pos > best_from_pos:
                best_to_pos -= 1
            # Insert at best_to_pos
            tour = np.insert(tour, best_to_pos, best_city)
            improved = True
            new_dist = best_dist + best_delta
            if new_dist < best_dist:
                best_dist = new_dist
                report_best_tour(tour)
            best_dist = new_dist

    return tour