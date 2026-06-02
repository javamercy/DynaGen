import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n)
    # nearest neighbor from each start, keep best
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
    
    # improvement loop
    for _ in range(50):
        # 2-opt local search (first improvement)
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a = tour[i]
                    b = tour[(i+1)%n]
                    c = tour[j]
                    d = tour[(j+1)%n]
                    if distance_matrix[a][b] + distance_matrix[c][d] > distance_matrix[a][c] + distance_matrix[b][d]:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        dist = sum(distance_matrix[tour[k-1]][tour[k]] for k in range(n))
                        if dist < best_dist:
                            best_dist = dist
                            best_tour = tour[:]
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved:
                    break
        # double-bridge kick
        # choose 4 distinct indices, ensure segments have at least 2 nodes
        if n < 8:
            # fallback: random 2-opt swap
            i = np.random.randint(n)
            j = (i + np.random.randint(2, n-2)) % n
            tour[i+1:j+1] = reversed(tour[i+1:j+1])
            dist = sum(distance_matrix[tour[k-1]][tour[k]] for k in range(n))
            if dist < best_dist:
                best_dist = dist
                best_tour = tour[:]
                report_best_tour(best_tour)
        else:
            # double-bridge: split into 4 segments and reorder
            idx = sorted(np.random.choice(range(1, n), 4, replace=False))
            a, b, c, d = idx
            # segments: [0:a], [a:b], [b:c], [c:d], [d:n]
            # reorder as: [0:a] + [c:d] + [b:c] + [a:b] + [d:n]
            new_tour = tour[:a] + tour[c:d] + tour[b:c] + tour[a:b] + tour[d:]
            dist = sum(distance_matrix[new_tour[k-1]][new_tour[k]] for k in range(n))
            tour = new_tour
            if dist < best_dist:
                best_dist = dist
                best_tour = tour[:]
                report_best_tour(best_tour)
    return np.array(best_tour)