import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    best_dist = np.inf
    best_tour = None
    for _ in range(10):
        start = np.random.randint(n)
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        cur = start
        while unvisited:
            next_city = min(unvisited, key=lambda x: distance_matrix[cur, x])
            tour.append(next_city)
            unvisited.remove(next_city)
            cur = next_city
        tour = np.array(tour)
        
        def two_opt(t):
            improved = True
            while improved:
                improved = False
                for i in range(n-2):
                    for j in range(i+2, n):
                        if j - i == 1:
                            continue
                        a, b = t[i], t[(i+1)%n]
                        c, d = t[j], t[(j+1)%n]
                        if distance_matrix[a,c] + distance_matrix[b,d] < distance_matrix[a,b] + distance_matrix[c,d]:
                            t[i+1:j+1] = t[i+1:j+1][::-1]
                            improved = True
            return t
        
        tour = two_opt(tour)
        curr_dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
        if curr_dist < best_dist:
            best_dist = curr_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        
        no_improve = 0
        for _ in range(20):
            i, j = np.random.choice(n, 2, replace=False)
            new_tour = tour.copy()
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            new_tour = two_opt(new_tour)
            new_dist = sum(distance_matrix[new_tour[i], new_tour[(i+1)%n]] for i in range(n))
            if new_dist < best_dist:
                best_dist = new_dist
                best_tour = new_tour.copy()
                report_best_tour(best_tour)
                tour = new_tour
                no_improve = 0
            elif new_dist < curr_dist:
                tour = new_tour
                curr_dist = new_dist
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 10:
                    break
        final_dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
        if final_dist < best_dist:
            best_dist = final_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
    return best_tour