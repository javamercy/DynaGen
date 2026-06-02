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
    
    # Improvement loop
    for iteration in range(20):
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
        # Random kick (except last iteration)
        if iteration >= 19:
            break
        i = np.random.randint(n)
        possible_j = [j for j in range(n) if (j-i)%n not in (0,1,n-1)]
        if not possible_j:
            continue
        j = np.random.choice(possible_j)
        if i > j:
            i, j = j, i
        if j - i < 2:
            continue
        tour[i+1:j+1] = reversed(tour[i+1:j+1])
    return np.array(best_tour)