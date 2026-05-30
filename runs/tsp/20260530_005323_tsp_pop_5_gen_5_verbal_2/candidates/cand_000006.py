import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 1:
        return np.array([0])
    # Cheapest insertion construction
    tour = [0]
    unvisited = set(range(1, n))
    while unvisited:
        best_increase = np.inf
        best_city = None
        best_pos = None
        for city in unvisited:
            for pos in range(len(tour) + 1):
                prev = tour[pos - 1] if pos > 0 else tour[-1]
                nxt = tour[pos] if pos < len(tour) else tour[0]
                increase = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
                if increase < best_increase:
                    best_increase = increase
                    best_city = city
                    best_pos = pos
        tour.insert(best_pos, best_city)
        unvisited.remove(best_city)
    tour = np.array(tour)
    best_tour = tour.copy()
    best_len = np.sum(distance_matrix[tour, np.roll(tour, -1)])
    report_best_tour(best_tour)
    # 2-opt and 3-opt improvement
    improved = True
    while improved:
        improved = False
        # 2-opt
        for i in range(n):
            for j in range(i + 2, n):
                a, b = tour[i], tour[(i + 1) % n]
                c, d = tour[j], tour[(j + 1) % n]
                if distance_matrix[a, c] + distance_matrix[b, d] < distance_matrix[a, b] + distance_matrix[c, d]:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
        if improved:
            new_len = np.sum(distance_matrix[tour, np.roll(tour, -1)])
            if new_len < best_len:
                best_len = new_len
                best_tour = tour.copy()
                report_best_tour(best_tour)
            continue
        # 3-opt (one type)
        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                for k in range(j + 2, n):
                    a, b = tour[i], tour[(i + 1) % n]
                    c, d = tour[j], tour[(j + 1) % n]
                    e, f = tour[k], tour[(k + 1) % n]
                    d0 = distance_matrix[a, b] + distance_matrix[c, d] + distance_matrix[e, f]
                    d1 = distance_matrix[a, c] + distance_matrix[b, e] + distance_matrix[d, f]
                    if d1 < d0 - 1e-12:
                        i_next = i + 1
                        j_next = j + 1
                        k_next = k + 1
                        new_tour = np.concatenate([
                            tour[:i_next],
                            tour[j_next:k_next][::-1],
                            tour[i_next:j_next][::-1],
                            tour[k_next:]
                        ])
                        tour = new_tour
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            new_len = np.sum(distance_matrix[tour, np.roll(tour, -1)])
            if new_len < best_len:
                best_len = new_len
                best_tour = tour.copy()
                report_best_tour(best_tour)
    return best_tour