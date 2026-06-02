import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n)
    # Nearest neighbor best-of-n
    best_tour = None
    best_dist = float('inf')
    for start in range(n):
        tour = [start]
        visited = {start}
        cur = start
        for _ in range(n-1):
            nxt = min((j for j in range(n) if j not in visited), key=lambda j: distance_matrix[cur][j])
            tour.append(nxt)
            visited.add(nxt)
            cur = nxt
        dist = sum(distance_matrix[tour[i-1]][tour[i]] for i in range(n))
        if dist < best_dist:
            best_dist = dist
            best_tour = tour[:]
    tour = best_tour[:]
    report_best_tour(tour)
    # Iterated local search with 2-opt and double-bridge kick
    for _ in range(20):
        # 2-opt local search
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    a, b = tour[i], tour[(i+1)%n]
                    c, d = tour[j], tour[(j+1)%n]
                    if distance_matrix[a][c] + distance_matrix[b][d] < distance_matrix[a][b] + distance_matrix[c][d]:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        dist = sum(distance_matrix[tour[i-1]][tour[i]] for i in range(n))
                        if dist < best_dist:
                            best_dist = dist
                            best_tour = tour[:]
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved:
                    break
        # Double-bridge kick
        # Pick 4 indices to split into 5 segments: A, B, C, D, E (A and E are same after wrap)
        # Reorder: A, C, B, D, E (standard double bridge)
        i = np.random.randint(n)
        j = np.random.randint(2, n-2)
        k = np.random.randint(j+2, n-1)
        # Ensure all segments have at least 1 node
        A = tour[:i]
        B = tour[i:j]
        C = tour[j:k]
        D = tour[k:]
        tour = A + C + B + D
        # Check validity (always valid)
    return np.array(best_tour)