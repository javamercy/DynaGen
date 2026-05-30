import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        return np.array([0])
    if n == 2:
        return np.array([0, 1])
    # farthest insertion construction
    max_dist = -1
    start_pair = (0, 1)
    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i][j] > max_dist:
                max_dist = distance_matrix[i][j]
                start_pair = (i, j)
    tour = [start_pair[0], start_pair[1]]
    unvisited = set(range(n)) - set(tour)
    while unvisited:
        best_node = None
        best_increase = -1
        best_pos = None
        for node in unvisited:
            min_increase = float('inf')
            min_pos = None
            m = len(tour)
            for i in range(m):
                j = (i + 1) % m
                inc = distance_matrix[tour[i]][node] + distance_matrix[node][tour[j]] - distance_matrix[tour[i]][tour[j]]
                if inc < min_increase:
                    min_increase = inc
                    min_pos = j
            if min_increase > best_increase:
                best_increase = min_increase
                best_node = node
                best_pos = min_pos
        tour.insert(best_pos, best_node)
        unvisited.remove(best_node)
    tour = np.array(tour)
    def _tour_length(t):
        total = 0.0
        for k in range(n):
            total += distance_matrix[t[k]][t[(k+1) % n]]
        return total
    best_dist = _tour_length(tour)
    report_best_tour(tour.copy())
    # VND
    improved = True
    while improved:
        improved = False
        # 2-opt first improvement
        inner_improved = True
        while inner_improved:
            inner_improved = False
            for i in range(n-2):
                for j in range(i+2, n-1):
                    a = tour[i]
                    b = tour[i+1]
                    c = tour[j]
                    d = tour[(j+1) % n]
                    delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                    if delta < -1e-12:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        new_dist = _tour_length(tour)
                        if new_dist < best_dist - 1e-12:
                            best_dist = new_dist
                            report_best_tour(tour.copy())
                        inner_improved = True
                        break
                if inner_improved:
                    break
        # Or-opt first improvement using delta
        for remove_idx in range(n):
            node = tour[remove_idx]
            # best insertion position for this node, excluding its current position
            best_inc = float('inf')
            best_pos = -1
            # consider insertion after each position (including at beginning)
            for pos in range(n):
                if pos == remove_idx or pos == (remove_idx - 1) % n or pos == remove_idx % n:
                    # skip positions that would leave tour unchanged
                    continue
                # delta if we remove node and insert after pos
                # handle wrapping correctly
                prev = tour[pos]
                next_node = tour[(pos+1) % n]
                # cost change: - edge (prev, next) + edge (prev, node) + edge (node, next)
                # plus removal of node's old edges
                old_prev = tour[(remove_idx-1) % n]
                old_next = tour[(remove_idx+1) % n]
                old_edges = distance_matrix[old_prev][node] + distance_matrix[node][old_next]
                new_edges = distance_matrix[prev][node] + distance_matrix[node][next_node]
                inc = new_edges - old_edges
                # Note: if removing node, the edge (old_prev, old_next) becomes new, so we must account for that
                # Actually the correct delta: remove node and its two incident edges, add edge between old_prev and old_next
                # Then insert node between prev and next, adding two new edges and removing one edge (prev-next)
                # So total delta = (distance_matrix[old_prev][old_next] - old_edges) + (new_edges - distance_matrix[prev][next_node])
                # Let's compute carefully:
                # After removal, the tour loses two edges and gains one edge (old_prev, old_next).
                # After insertion, loses one edge (prev, next) and gains two edges (prev, node) and (node, next).
                # So net change = (distance_matrix[old_prev][old_next] - old_edges) + (new_edges - distance_matrix[prev][next_node])
                inc = (distance_matrix[old_prev][old_next] - old_edges) + (new_edges - distance_matrix[prev][next_node])
                # This inc is the change in total distance (positive means worse, negative better)
                if inc < best_inc:
                    best_inc = inc
                    best_pos = pos
            if best_inc < -1e-12:
                # apply move: remove node at remove_idx, then insert after best_pos (adjusting indices)
                # Remove first
                new_tour = np.delete(tour, remove_idx)
                # Insert after best_pos
                if best_pos >= remove_idx:
                    # because we removed, best_pos shifts? Actually after removal, indices after remove_idx shift left by 1.
                    # If best_pos >= remove_idx, the insertion index in new_tour is best_pos (since the element at best_pos originally is now at best_pos-1? Let's handle carefully)
                    # Better to insert at position (best_pos+1) in new_tour if best_pos is the index in original before removal.
                    # Simpler: create a list and insert
                    tour_list = list(tour)
                    node = tour_list.pop(remove_idx)
                    if best_pos >= remove_idx:
                        # after pop, indices shift left; the original position best_pos is now at best_pos-1
                        actual_pos = best_pos  # we want to insert after the element that was at original best_pos
                        # Actually we want to insert after best_pos, so insertion index = best_pos+1, but after pop, the element that was at best_pos+1 becomes at best_pos? Let's test.
                        # Suppose tour = [0,1,2,3], remove_idx=1 (node=1), best_pos=2 (after 2). After removal, tour becomes [0,2,3]. We want to insert after 2, which is at index 1 in new list? Actually after removal, the element originally at index 2 (value 2) is now at index 1. We want to insert after it, so index 2. So insertion index = best_pos if best_pos >= remove_idx? That would be 2, correct. If best_pos < remove_idx, insertion index = best_pos+1? Let's test: remove_idx=2, best_pos=0. After removal of index 2, tour becomes [0,1,3]. We want to insert after element at original index 0 (value 0), which is at new index 0. Insert at index 1. So insertion index = best_pos+1 when best_pos < remove_idx. So logic:
                        if best_pos >= remove_idx:
                            insert_pos = best_pos  # after pop, the element that was at best_pos is now at best_pos, so insert after it means at best_pos+1? Wait, we want to insert after the element, so the insertion index is the position after the element's current index. After pop, the element's index is best_pos (if best_pos >= remove_idx, it shifted left by 1? Actually no: if best_pos > remove_idx, the element's index decreases by 1. For example, original indices: 0,1,2,3. Remove index 1. New indices: 0 (original 0), 1 (original 2), 2 (original 3). So original index 2 becomes new index 1. So to insert after original index 2, we need to insert at new index 2. So insertion index = best_pos (since best_pos = 2, and new index of that element is 1, so after it is index 2). So if best_pos >= remove_idx, insert_pos = best_pos. If best_pos < remove_idx, original index best_pos stays same, so insertion after it is at best_pos+1. So correct.
                        insert_pos = best_pos if best_pos >= remove_idx else best_pos + 1
                    else:
                        insert_pos = best_pos + 1
                    tour_list.insert(insert_pos, node)
                    tour = np.array(tour_list)
                    new_dist = _tour_length(tour)
                    if new_dist < best_dist - 1e-12:
                        best_dist = new_dist
                        report_best_tour(tour.copy())
                    improved = True
                    break  # apply first improvement
        # break condition: if no improvement in both 2-opt and Or-opt, while loop ends
    return tour