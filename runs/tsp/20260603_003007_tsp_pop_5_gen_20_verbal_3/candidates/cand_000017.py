import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour
    best_tour = None
    best_dist = np.inf
    # try each city as start (multi-start)
    for start in range(min(n, 5)):  # limit to 5 starts for speed
        # nearest neighbor construction
        unvisited = list(range(n))
        tour = [start]
        unvisited.remove(start)
        while unvisited:
            last = tour[-1]
            # find nearest unvisited
            min_dist = np.inf
            nearest = -1
            for node in unvisited:
                d = distance_matrix[last, node]
                if d < min_dist:
                    min_dist = d
                    nearest = node
            tour.append(nearest)
            unvisited.remove(nearest)
        tour_arr = np.array(tour, dtype=np.int32)
        # 2-opt best-improvement
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            # precompute distances for speed
            for i in range(n):
                i1 = (i + 1) % n
                for j in range(i + 2, n):
                    j1 = (j + 1) % n
                    delta = (distance_matrix[tour_arr[i], tour_arr[j]] +
                             distance_matrix[tour_arr[i1], tour_arr[j1]] -
                             distance_matrix[tour_arr[i], tour_arr[i1]] -
                             distance_matrix[tour_arr[j], tour_arr[j1]])
                    if delta < best_delta - 1e-12:
                        best_delta = delta
                        best_i = i
                        best_j = j
            if best_delta < -1e-12:
                i, j = best_i, best_j
                # reverse segment i+1..j
                tour_arr[i+1:j+1] = tour_arr[i+1:j+1][::-1]
                improved = True
                dist = compute_dist(tour_arr, distance_matrix)
                if dist < best_dist:
                    best_dist = dist
                    best_tour = tour_arr.copy()
                    report_best_tour(best_tour)
        # after 2-opt, evaluate tour
        dist = compute_dist(tour_arr, distance_matrix)
        if dist < best_dist:
            best_dist = dist
            best_tour = tour_arr.copy()
            report_best_tour(best_tour)
    return best_tour

def compute_dist(tour, mat):
    n = len(tour)
    total = mat[tour[-1], tour[0]]
    for i in range(n-1):
        total += mat[tour[i], tour[i+1]]
    return total