import numpy as np
def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # farthest-insertion construction
    tour = [0, int(np.argmax(distance_matrix[0]))]
    in_tour = set(tour)
    while len(tour) < n:
        best_node = -1
        best_dist = -1.0
        for node in range(n):
            if node not in in_tour:
                min_dist = min(distance_matrix[node][t] for t in tour)
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_node = node
        best_pos = -1
        best_increase = float('inf')
        for i in range(len(tour)):
            a = tour[i]
            b = tour[(i+1)%len(tour)]
            increase = distance_matrix[a][best_node] + distance_matrix[best_node][b] - distance_matrix[a][b]
            if increase < best_increase:
                best_increase = increase
                best_pos = i+1
        tour.insert(best_pos, best_node)
        in_tour.add(best_node)
    tour = np.array(tour)
    best_dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    report_best_tour(tour)
    # 2-opt local search
    def two_opt(tour_list):
        improved = True
        while improved:
            improved = False
            for i in range(n-1):
                for j in range(i+1, n):
                    if j-i == 1 or (i==0 and j==n-1):
                        continue
                    a, b = tour_list[i], tour_list[(i+1)%n]
                    c, d = tour_list[j], tour_list[(j+1)%n]
                    delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                    if delta < -1e-10:
                        tour_list[i+1:j+1] = reversed(tour_list[i+1:j+1])
                        improved = True
        return tour_list
    tour = two_opt(list(tour))
    best_tour = np.array(tour)
    best_dist = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour)
    # restart with double-bridge kick and random swap
    for _ in range(30):
        tour_list = list(best_tour)
        # double-bridge kick
        p1 = np.random.randint(0, n//4)
        p2 = np.random.randint(p1+1, n//2)
        p3 = np.random.randint(p2+1, 3*n//4)
        p4 = np.random.randint(p3+1, n-1)
        seg1 = tour_list[p1:p2]
        seg2 = tour_list[p2:p3]
        seg3 = tour_list[p3:p4]
        seg4 = tour_list[p4:] + tour_list[:p1]
        new_tour = seg1 + seg3 + seg2 + seg4
        if len(set(new_tour)) != n:
            continue
        # extra diversification: random swap
        i, j = np.random.randint(0, n, size=2)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        new_tour = two_opt(new_tour)
        new_dist = sum(distance_matrix[new_tour[i], new_tour[(i+1)%n]] for i in range(n))
        if new_dist < best_dist - 1e-10:
            best_tour = np.array(new_tour)
            best_dist = new_dist
            report_best_tour(best_tour)
    return best_tour