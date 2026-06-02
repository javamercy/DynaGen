import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n)

    # best-of-n nearest neighbor
    best_tour = None
    best_len = float('inf')
    for start in range(n):
        tour = [start]
        visited = {start}
        cur = start
        for _ in range(n-1):
            next_node = min([j for j in range(n) if j not in visited], key=lambda j: distance_matrix[cur][j])
            tour.append(next_node)
            visited.add(next_node)
            cur = next_node
        length = sum(distance_matrix[tour[i-1]][tour[i]] for i in range(n))
        if length < best_len:
            best_len = length
            best_tour = tour[:]
            report_best_tour(best_tour)

    tour = best_tour[:]
    # improvement loop
    for iteration in range(20):
        # 2-opt local search
        improved = True
        while improved:
            improved = False
            for i in range(n-1):
                for j in range(i+2, n):
                    if j - i == 1:
                        continue
                    a, b = tour[i], tour[i+1]
                    c, d = tour[j], tour[(j+1)%n]
                    delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                    if delta < -1e-10:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        if length + delta < best_len - 1e-10:
                            best_len = length + delta
                            best_tour = tour[:]
                            report_best_tour(best_tour)
                        length += delta
                        improved = True
                        break
                if improved:
                    break
        if iteration == 19:
            break
        # double-bridge kick
        while True:
            i = np.random.randint(1, n)
            j = np.random.randint(1, n)
            k = np.random.randint(1, n)
            if i < j < k and (j-i) > 1 and (k-j) > 1 and (n-k) > 1:
                break
        # segments: A[0:i], B[i:j], C[j:k], D[k:]
        tour = tour[:i] + tour[k:] + tour[j:k] + tour[i:j]
        length = sum(distance_matrix[tour[p-1]][tour[p]] for p in range(n))

    return np.array(best_tour)