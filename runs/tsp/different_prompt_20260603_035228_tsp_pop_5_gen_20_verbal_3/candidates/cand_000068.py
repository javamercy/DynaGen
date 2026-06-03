import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n, dtype=int)
        report_best_tour(tour)
        return tour

    def total_dist(t):
        idx = np.array(t, dtype=int)
        return distance_matrix[idx[-1], idx[0]] + np.sum(distance_matrix[idx[:-1], idx[1:]])

    def first_improvement_2opt(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == n - 1:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[0]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[0]])
                    else:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[j+1]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[j+1]])
                    if delta > 1e-12:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return tour

    def double_bridge(tour):
        cuts = sorted(np.random.choice(range(1, n), 3, replace=False))
        seg0 = tour[:cuts[0]]
        seg1 = tour[cuts[0]:cuts[1]]
        seg2 = tour[cuts[1]:cuts[2]]
        seg3 = tour[cuts[2]:]
        return np.concatenate([seg0, seg2, seg1, seg3])

    def farthest_insertion():
        tour = [np.random.randint(n)]
        visited = [False] * n
        visited[tour[0]] = True
        while len(tour) < n:
            best_dist = -1.0
            best_node = None
            for i in range(n):
                if not visited[i]:
                    min_to_tour = min(distance_matrix[i, tour[j]] for j in range(len(tour)))
                    if min_to_tour > best_dist:
                        best_dist = min_to_tour
                        best_node = i
            # Insert best_node in the best position (minimum increase)
            best_increase = float('inf')
            best_pos = 0
            for pos in range(len(tour)):
                # Insert between tour[pos] and tour[(pos+1)%len(tour)]
                a = tour[pos]
                b = tour[(pos+1) % len(tour)]
                increase = distance_matrix[a, best_node] + distance_matrix[best_node, b] - distance_matrix[a, b]
                if increase < best_increase:
                    best_increase = increase
                    best_pos = pos + 1
            tour.insert(best_pos, best_node)
            visited[best_node] = True
        return np.array(tour, dtype=int)

    best_tour = None
    best_dist = float('inf')
    num_restarts = 5
    max_cycles = 20
    stall_limit = 5

    for _ in range(num_restarts):
        tour = farthest_insertion()
        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)

        no_improve = 0
        for cycle in range(max_cycles):
            tour = first_improvement_2opt(tour)
            cur_dist = total_dist(tour)
            if cur_dist < best_dist - 1e-12:
                best_dist = cur_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
                no_improve = 0
            else:
                no_improve += 1

            if cycle == max_cycles - 1:
                break

            if no_improve >= stall_limit:
                # Restart with a new random construction
                tour = farthest_insertion()
                no_improve = 0
            else:
                if no_improve == 0:
                    seg_len = np.random.randint(2, max(3, n//4 + 2))
                    i = np.random.randint(0, n - seg_len)
                    tour[i:i+seg_len] = tour[i:i+seg_len][::-1]
                else:
                    tour = double_bridge(tour)

    return best_tour