import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        if n == 2:
            report_best_tour(tour)
        return tour
    rng = random.Random(seed)
    # Candidate list: for each node, sorted indices by distance (excluding self)
    candidate_size = 30 if n < 80 else 20
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        # Exclude self (distance 0), argsort gives ascending order
        order = np.argsort(dists)
        order = order[order != i]  # remove self
        if len(order) > candidate_size:
            order = order[:candidate_size]
        candidates.append(list(order))
    
    # Greedy nearest neighbor with random start
    def greedy_tour():
        start = rng.randint(0, n-1)
        visited = set([start])
        tour = [start]
        current = start
        while len(visited) < n:
            # Find nearest unvisited neighbor
            dists = distance_matrix[current]
            best = None
            best_dist = float('inf')
            for j in range(n):
                if j not in visited and dists[j] < best_dist:
                    best = j
                    best_dist = dists[j]
            visited.add(best)
            tour.append(best)
            current = best
        return np.array(tour, dtype=np.int32)
    
    tour = greedy_tour()
    report_best_tour(tour)
    
    # Helper functions
    def get_pos(tour):
        pos = np.empty(n, dtype=np.int32)
        for idx, node in enumerate(tour):
            pos[node] = idx
        return pos
    
    def tour_length(tour):
        total = 0
        for i in range(n):
            total += distance_matrix[tour[i]][tour[(i+1)%n]]
        return total
    
    current_dist = tour_length(tour)
    pos = get_pos(tour)
    
    # Precompute all 2-opt candidate pairs (i, j) where j > i and candidate of tour[i]
    def generate_2opt_pairs(tour, pos):
        pairs = []
        for i in range(n):
            node = tour[i]
            for cand in candidates[node]:
                j = pos[cand]
                if j <= i:
                    continue
                # Avoid adjacent edges: i+1 should not be j, and j+1 should not be i (mod n)
                if (i+1)%n == j or (j+1)%n == i:
                    continue
                pairs.append((i, j))
        return pairs
    
    # Generate relocation pairs (u, v) where v is candidate of u and v != u, and moving u after v is non-trivial
    def generate_reloc_pairs(tour, pos):
        pairs = []
        for u in range(n):
            i = pos[u]
            for v in candidates[u]:
                if v == u:
                    continue
                k = pos[v]
                # Cannot move to same position or adjacent (mod n) to avoid degenerate
                # Actually moving to after v is fine even if adjacent, but we skip if k==i or (k+1)%n == i?
                # We'll allow all, the delta calculation handles
                pairs.append((u, v))
        return pairs
    
    evals = 0
    improved = True
    while evals < budget and improved:
        improved = False
        # First 2-opt phase
        pairs = generate_2opt_pairs(tour, pos)
        rng.shuffle(pairs)
        for (i, j) in pairs:
            if evals >= budget:
                break
            a = tour[i]
            b = tour[(i+1)%n]
            c = tour[j]
            d = tour[(j+1)%n]
            old = distance_matrix[a][b] + distance_matrix[c][d]
            new = distance_matrix[a][c] + distance_matrix[b][d]
            delta = new - old
            evals += 1
            if delta < -1e-12:
                # apply 2-opt
                if i < j:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                else: # wrap around: reverse from i+1 to end and from start to j
                    segment = np.concatenate((tour[i+1:], tour[:j+1]))
                    segment = segment[::-1]
                    tour[i+1:] = segment[:n-i-1]
                    tour[:j+1] = segment[n-i-1:]
                pos = get_pos(tour)
                current_dist += delta
                report_best_tour(tour)
                improved = True
                break
        if improved:
            continue
        # Relocation phase
        pairs = generate_reloc_pairs(tour, pos)
        rng.shuffle(pairs)
        for (u, v) in pairs:
            if evals >= budget:
                break
            i = pos[u]
            k = pos[v]
            if i == k:
                continue
            prev_u = tour[(i-1)%n]
            next_u = tour[(i+1)%n]
            prev_v = tour[k]
            next_v = tour[(k+1)%n]
            old_edges = distance_matrix[prev_u][u] + distance_matrix[u][next_u] + distance_matrix[prev_v][next_v]
            new_edges = distance_matrix[prev_u][next_u] + distance_matrix[prev_v][u] + distance_matrix[u][next_v]
            delta = new_edges - old_edges
            evals += 1
            if delta < -1e-12:
                # apply relocation: remove u from its position and insert after v
                # Build new tour list
                new_tour = []
                for idx in range(n):
                    if idx == i:
                        continue
                    new_tour.append(tour[idx])
                # Now insert u after v in new list; find new index of v
                # Since removal shifted indices, we need to find where v is now
                # Using the original indices, we can compute new index
                # Simpler: construct full list manually
                tour_list = tour.tolist()
                # remove u
                tour_list.remove(u)
                # get index of v (since u removed, indices shifted)
                idx_v = tour_list.index(v)
                # insert u after v
                tour_list.insert(idx_v+1, u)
                tour = np.array(tour_list, dtype=np.int32)
                pos = get_pos(tour)
                current_dist += delta
                report_best_tour(tour)
                improved = True
                break
    return tour