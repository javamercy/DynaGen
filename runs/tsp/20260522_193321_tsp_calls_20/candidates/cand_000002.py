import numpy as np, random, math, heapq, itertools, collections, time

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = list(range(n))
        try:
            report_best_tour(tour)
        except:
            pass
        return np.array(tour)
    random.seed(seed)
    # Construction: regret-2 insertion
    start = random.randrange(n)
    tour = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    while unvisited:
        best_city = None
        best_regret = -1
        best_pos = None
        best_cost = None
        for city in unvisited:
            min_cost = float('inf')
            second_min = float('inf')
            best_idx = -1
            for idx in range(len(tour)):
                if idx == len(tour)-1:
                    cost = distance_matrix[tour[idx]][city] + distance_matrix[city][tour[0]] - distance_matrix[tour[idx]][tour[0]]
                else:
                    cost = distance_matrix[tour[idx]][city] + distance_matrix[city][tour[idx+1]] - distance_matrix[tour[idx]][tour[idx+1]]
                if cost < min_cost:
                    second_min = min_cost
                    min_cost = cost
                    best_idx = idx
                elif cost < second_min:
                    second_min = cost
            regret = second_min - min_cost
            if regret > best_regret:
                best_regret = regret
                best_city = city
                best_pos = best_idx
                best_cost = min_cost
        tour.insert(best_pos+1, best_city)
        unvisited.remove(best_city)
    try:
        report_best_tour(np.array(tour))
    except:
        pass
    # Improvement: 2-opt with budget-limited iterations
    max_iter = budget // 10
    if max_iter > 1000:
        max_iter = 1000
    improvement = True
    iteration = 0
    while improvement and iteration < max_iter:
        improvement = False
        nc = len(tour)
        for i in range(nc-1):
            for j in range(i+2, nc):
                if j - i == 1:
                    continue
                a = tour[i]
                b = tour[(i+1)%nc]
                c = tour[j]
                d = tour[(j+1)%nc]
                if distance_matrix[a][c] + distance_matrix[b][d] < distance_matrix[a][b] + distance_matrix[c][d]:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improvement = True
                    try:
                        report_best_tour(np.array(tour))
                    except:
                        pass
        iteration += 1
    return np.array(tour)