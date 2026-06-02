import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n)
    # nearest neighbor initial tour (best of all starts)
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
    
    # improvement: 2-opt + random kick
    for iteration in range(20):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                # rotate so that i is at front
                rotated = tour[i:] + tour[:i]
                a = rotated[0]
                b = rotated[1]
                for offset in range(2, n):
                    j = offset
                    c = rotated[j]
                    d = rotated[(j+1)%n]
                    current_len = distance_matrix[a][b] + distance_matrix[c][d]
                    new_len = distance_matrix[a][c] + distance_matrix[b][d]
                    if new_len < current_len:
                        # apply 2-opt reverse segment
                        rotated[1:j+1] = reversed(rotated[1:j+1])
                        tour = rotated[i:] + rotated[:i]
                        dist = sum(distance_matrix[tour[i-1]][tour[i]] for i in range(n))
                        if dist < best_dist:
                            best_dist = dist
                            best_tour = tour[:]
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved:
                    break
        # random kick (except last iteration)
        if iteration >= 19:
            break
        # select a random non-adjacent edge pair
        i = np.random.randint(n)
        possible = [k for k in range(n) if (k-i)%n not in (0,1,n-1)]
        if not possible:
            continue
        j = np.random.choice(possible)
        # apply 2-opt move regardless of gain
        rotated = tour[i:] + tour[:i]
        offset = (j - i) % n
        if offset < 2 or offset >= n:
            continue
        rotated[1:offset+1] = reversed(rotated[1:offset+1])
        tour = rotated[i:] + rotated[:i]
    return np.array(best_tour)