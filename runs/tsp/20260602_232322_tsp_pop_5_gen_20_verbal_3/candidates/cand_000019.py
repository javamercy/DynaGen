import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n)
    best_tour = None
    best_dist = float('inf')
    
    def tour_length(t):
        return sum(distance_matrix[t[i-1]][t[i]] for i in range(n))
    
    def report(t):
        d = tour_length(t)
        nonlocal best_dist, best_tour
        if d < best_dist:
            best_dist = d
            best_tour = t[:]
            report_best_tour(t)
    
    # Multi-start nearest neighbor
    for start in range(min(5, n)):  # limit restarts for speed
        tour = [start]
        visited = {start}
        current = start
        for _ in range(n-1):
            next_node = min((j for j in range(n) if j not in visited), key=lambda j: distance_matrix[current][j])
            tour.append(next_node)
            visited.add(next_node)
            current = next_node
        report(tour)
        
        # 2-opt improvement
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    if j == n-1 and i == 0:
                        continue
                    a = tour[i]
                    b = tour[(i+1)%n]
                    c = tour[j]
                    d = tour[(j+1)%n]
                    if distance_matrix[a][b] + distance_matrix[c][d] > distance_matrix[a][c] + distance_matrix[b][d]:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
                        report(tour)
                        break
                if improved:
                    break
        
        # Double bridge kick (random 4-opt) to escape local optimum
        if n >= 8:
            for _ in range(10):  # try a few kicks
                # partition tour into 4 segments
                indices = sorted(np.random.choice(range(1, n-1), size=3, replace=False))
                a, b, c, d = indices[0], indices[1], indices[2]
                # segments: [0:a], [a:b], [b:c], [c:d] (end of tour treated as wrap to start?)
                # double bridge: recombine as 1-3-2-4
                new_tour = tour[:a] + tour[b:c] + tour[a:b] + tour[c:]
                if tour_length(new_tour) < tour_length(tour):
                    tour = new_tour
                    report(tour)
                    # then run 2-opt again
                    improved = True
                    while improved:
                        improved = False
                        for i in range(n):
                            for j in range(i+2, n):
                                if j == n-1 and i == 0:
                                    continue
                                a_ = tour[i]
                                b_ = tour[(i+1)%n]
                                c_ = tour[j]
                                d_ = tour[(j+1)%n]
                                if distance_matrix[a_][b_] + distance_matrix[c_][d_] > distance_matrix[a_][c_] + distance_matrix[b_][d_]:
                                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                                    improved = True
                                    report(tour)
                                    break
                            if improved:
                                break
                    break  # only apply one successful kick per restart
    return np.array(best_tour)