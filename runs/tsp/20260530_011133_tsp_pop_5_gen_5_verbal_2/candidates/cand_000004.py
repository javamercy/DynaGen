import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    # nearest neighbor initialization
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        next_city = min(unvisited, key=lambda c: distance_matrix[current][c])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    best = np.array(tour)
    best_cost = sum(distance_matrix[tour[i]][tour[(i+1)%n]] for i in range(n))
    try:
        report_best_tour(best)
    except:
        pass
    improved = True
    while improved:
        improved = False
        # 2-opt
        for i in range(n-1):
            for j in range(i+2, n):
                a, b = tour[i], tour[(i+1)%n]
                c, d = tour[j], tour[(j+1)%n]
                delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                if delta < -1e-9:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    best_cost += delta
                    best = np.array(tour)
                    improved = True
                    try:
                        report_best_tour(best)
                    except:
                        pass
        # 3-opt (one variant: reverse two segments sequentially)
        for i in range(n-2):
            for j in range(i+2, n-1):
                for k in range(j+2, n):
                    # original edges: (i,i+1), (j,j+1), (k,k+1)
                    a, b, c, d, e, f = tour[i], tour[i+1], tour[j], tour[j+1], tour[k], tour[(k+1)%n]
                    # consider new edges: (a,c), (b,e), (d,f)
                    delta = (distance_matrix[a][c] + distance_matrix[b][e] + distance_matrix[d][f] -
                             distance_matrix[a][b] - distance_matrix[c][d] - distance_matrix[e][f])
                    if delta < -1e-9:
                        # apply: reverse segment i+1..j, then j+1..k
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        tour[j+1:k+1] = reversed(tour[j+1:k+1])
                        best_cost += delta
                        best = np.array(tour)
                        improved = True
                        try:
                            report_best_tour(best)
                        except:
                            pass
    return best