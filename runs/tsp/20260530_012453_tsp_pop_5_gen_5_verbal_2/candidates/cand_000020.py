import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    
    # farthest-insertion construction
    tour = [0, np.argmax(distance_matrix[0])]
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
    
    best_tour = np.array(tour)
    report_best_tour(best_tour)
    
    # 2-opt improvement
    def two_opt(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n-1):
                for j in range(i+1, n):
                    if j-i == 1 or (i==0 and j==n-1):
                        continue
                    a = tour[i]
                    b = tour[(i+1)%n]
                    c = tour[j]
                    d = tour[(j+1)%n]
                    delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                    if delta < -1e-10:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
        return np.array(tour)
    
    best_tour = two_opt(tour)
    report_best_tour(best_tour)
    
    # Restart loop with double-bridge kick
    for _ in range(5):
        # double-bridge kick
        indices = np.random.choice(n, 4, replace=False)
        indices.sort()
        a, b, c, d = indices
        # Split tour into four segments: [0:a], [a:b], [b:c], [c:d], [d:] and reassemble in different order
        new_tour = np.concatenate([tour[:a], tour[c:d], tour[b:c], tour[a:b], tour[d:]])
        tour = new_tour.tolist()
        # Apply 2-opt
        tour = two_opt(tour)
        # Update best if improved
        if np.sum(distance_matrix[tour, np.roll(tour, -1)]) < np.sum(distance_matrix[best_tour, np.roll(best_tour, -1)]):
            best_tour = tour
            report_best_tour(best_tour)
        # Reset tour to best for next restart
        tour = best_tour.tolist()
    
    return best_tour