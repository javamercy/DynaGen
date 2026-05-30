import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n == 1:
        tour = np.array([0])
        report_best_tour(tour)
        return tour
    # farthest insertion construction
    tour = [0]
    unvisited = set(range(1, n))
    while unvisited:
        best_node = None
        best_min_dist = -1
        for v in unvisited:
            min_dist = min(distance_matrix[v, t] for t in tour)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_node = v
        best_increase = np.inf
        best_pos = 0
        for i in range(len(tour)):
            prev = tour[i]
            nxt = tour[(i+1) % len(tour)]
            increase = distance_matrix[prev, best_node] + distance_matrix[best_node, nxt] - distance_matrix[prev, nxt]
            if increase < best_increase:
                best_increase = increase
                best_pos = i+1
        tour.insert(best_pos, best_node)
        unvisited.remove(best_node)
    tour = np.array(tour, dtype=np.int32)
    def tour_cost(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))
    best_tour = tour.copy()
    best_cost = tour_cost(best_tour)
    report_best_tour(best_tour)
    # 3-opt improvement with perturbation
    max_no_improve = 10
    no_improve = 0
    perturbation_limit = 5
    perturb_count = 0
    improved = True
    while improved or (perturb_count < perturbation_limit):
        improved = False
        i = 0
        while i < n and not improved:
            j = i + 2
            while j < n - 1 and not improved:
                k = j + 2
                while k < n and not improved:
                    a = tour[i]
                    b = tour[(i+1)%n]
                    c = tour[j]
                    d = tour[(j+1)%n]
                    e = tour[k]
                    f = tour[(k+1)%n]
                    cur = distance_matrix[a,b] + distance_matrix[c,d] + distance_matrix[e,f]
                    # type 1: reverse i+1..j
                    new1 = distance_matrix[a,d] + distance_matrix[c,b] + distance_matrix[e,f]
                    if new1 < cur:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
                        break
                    # type 2: reverse j+1..k
                    new2 = distance_matrix[a,b] + distance_matrix[c,f] + distance_matrix[e,d]
                    if new2 < cur:
                        tour[j+1:k+1] = tour[j+1:k+1][::-1]
                        improved = True
                        break
                    # type 3: reverse both
                    new3 = distance_matrix[a,d] + distance_matrix[c,f] + distance_matrix[e,b]
                    if new3 < cur:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        tour[j+1:k+1] = tour[j+1:k+1][::-1]
                        improved = True
                        break
                    k += 1
                j += 1
            i += 1
        if improved:
            no_improve = 0
            cost = tour_cost(tour)
            if cost < best_cost:
                best_cost = cost
                best_tour = tour.copy()
                report_best_tour(best_tour)
        else:
            no_improve += 1
            if no_improve >= max_no_improve and perturb_count < perturbation_limit:
                # random swap perturbation
                i, j = np.random.choice(n, 2, replace=False)
                tour[i], tour[j] = tour[j], tour[i]
                cost = tour_cost(tour)
                if cost < best_cost:
                    best_cost = cost
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                perturb_count += 1
                no_improve = 0
                improved = True
            else:
                break
    return best_tour