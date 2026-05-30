import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    best_tour = None
    best_cost = np.inf
    for restart in range(10):
        # initial tour
        if restart == 0:
            # nearest neighbor
            tour = [0]
            unvisited = set(range(1, n))
            curr = 0
            while unvisited:
                nxt = min(unvisited, key=lambda c: distance_matrix[curr, c])
                tour.append(nxt)
                unvisited.remove(nxt)
                curr = nxt
            tour = np.array(tour, dtype=np.int32)
        else:
            # perturb best_tour by a random 2-opt move
            tour = best_tour.copy()
            i = np.random.randint(0, n)
            j = (i + np.random.randint(2, n-1)) % n
            if i > j:
                i, j = j, i
            tour[i+1:j+1] = tour[i+1:j+1][::-1]
        cost = 0.0
        for k in range(n):
            cost += distance_matrix[tour[k], tour[(k+1)%n]]
        if cost < best_cost:
            best_cost = cost
            best_tour = tour.copy()
            report_best_tour(best_tour)
        improved = True
        while improved:
            improved = False
            # full 2-opt scan
            for i in range(n):
                for j in range(i+2, n):
                    a, b = tour[i], tour[(i+1)%n]
                    c, d = tour[j], tour[(j+1)%n]
                    if distance_matrix[a, c] + distance_matrix[b, d] < distance_matrix[a, b] + distance_matrix[c, d]:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        cost = 0.0
                        for k in range(n):
                            cost += distance_matrix[tour[k], tour[(k+1)%n]]
                        if cost < best_cost:
                            best_cost = cost
                            best_tour = tour.copy()
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved:
                    break
    return best_tour