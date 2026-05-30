import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    # nearest neighbor construction
    tour = [0]
    unvisited = set(range(1, n))
    curr = 0
    while unvisited:
        nxt = min(unvisited, key=lambda x: distance_matrix[curr, x])
        tour.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    tour = np.array(tour)
    
    def tour_len(t):
        return distance_matrix[t[-1], t[0]] + sum(distance_matrix[t[i], t[i+1]] for i in range(n-1))
    
    best_tour = tour.copy()
    best_len = tour_len(tour)
    try:
        report_best_tour(best_tour)
    except NameError:
        pass
    
    improved = True
    while improved:
        improved = False
        # 2-opt
        for i in range(n):
            i_next = (i+1) % n
            for j in range(i+2, n):
                if j == i_next:
                    continue
                j_next = (j+1) % n
                a, b = tour[i], tour[i_next]
                c, d = tour[j], tour[j_next]
                if distance_matrix[a, c] + distance_matrix[b, d] < distance_matrix[a, b] + distance_matrix[c, d]:
                    # reverse segment from i_next to j
                    if i_next < j:
                        tour = np.concatenate([tour[:i_next], tour[i_next:j+1][::-1], tour[j+1:]])
                    else:
                        # wrap around, need to handle differently
                        seg = np.concatenate([tour[i_next:], tour[:j+1]])
                        seg = seg[::-1]
                        tour = np.concatenate([seg[:len(tour)-i_next], seg[len(tour)-i_next:]])
                    improved = True
                    new_len = tour_len(tour)
                    if new_len < best_len:
                        best_len = new_len
                        best_tour = tour.copy()
                        try:
                            report_best_tour(best_tour)
                        except:
                            pass
        # 3-opt (only consider non-crossing cases that improve)
        for i in range(n):
            i_next = (i+1) % n
            for j in range(i+2, n-1):
                j_next = (j+1) % n
                for k in range(j+2, n):
                    if k == j_next:
                        continue
                    k_next = (k+1) % n
                    # There are 4 possible 3-opt moves. We'll check the most common one: reverse two segments
                    a, b = tour[i], tour[i_next]
                    c, d = tour[j], tour[j_next]
                    e, f = tour[k], tour[k_next]
                    # case 1: replace abc with a-c from b-e and d-f
                    d1 = distance_matrix[a, c] + distance_matrix[b, e] + distance_matrix[d, f]
                    d0 = distance_matrix[a, b] + distance_matrix[c, d] + distance_matrix[e, f]
                    if d1 < d0:
                        # apply: reverse segment i_next..j, then reverse j_next..k (or something)
                        # Simplified: we can reconstruct tour by taking segments
                        tour = np.concatenate([tour[:i_next], tour[j:k+1][::-1], tour[i_next:j+1][::-1], tour[k+1:]])
                        improved = True
                        new_len = tour_len(tour)
                        if new_len < best_len:
                            best_len = new_len
                            best_tour = tour.copy()
                            try:
                                report_best_tour(best_tour)
                            except:
                                pass
    # make sure to return best tour
    return best_tour