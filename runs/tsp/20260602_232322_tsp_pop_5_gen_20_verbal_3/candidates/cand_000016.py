import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n)
    
    # Nearest neighbor from each start, keep best
    best_tour = None
    best_dist = float('inf')
    for start in range(n):
        tour = [start]
        visited = {start}
        current = start
        for _ in range(n-1):
            next_node = min((j for j in range(n) if j not in visited), key=lambda j: distance_matrix[current][j])
            tour.append(next_node)
            visited.add(next_node)
            current = next_node
        dist = sum(distance_matrix[tour[i-1]][tour[i]] for i in range(n))
        if dist < best_dist:
            best_dist = dist
            best_tour = tour[:]
    tour = best_tour[:]
    report_best_tour(tour)
    
    # Improvement loop with double-bridge kicks
    for iteration in range(30):
        # 2-opt local search
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    if j - i == 1:
                        continue
                    a, b = tour[i], tour[i+1]
                    c, d = tour[j], tour[(j+1)%n]
                    delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                    if delta < -1e-10:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        dist = sum(distance_matrix[tour[k-1]][tour[k]] for k in range(n))
                        if dist < best_dist - 1e-10:
                            best_dist = dist
                            best_tour = tour[:]
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved:
                    break
        # Double-bridge kick (except last iteration)
        if iteration >= 29:
            break
        # Choose 4 random cut points and sort them
        cuts = sorted(np.random.choice(range(1, n), 4, replace=False))
        a, b, c, d = cuts[0], cuts[1], cuts[2], cuts[3]
        # Reorder segments: 1->3->2->4
        tour = tour[:a] + tour[b:c] + tour[a:b] + tour[c:d] + tour[d:]
    return np.array(best_tour)