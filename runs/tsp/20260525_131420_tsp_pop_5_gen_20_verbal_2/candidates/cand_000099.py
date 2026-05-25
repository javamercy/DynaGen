import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)

    k = min(n-1, 80)
    nn_list = []
    for i in range(n):
        order = np.argsort(distance_matrix[i])
        nn_list.append(order[1:k+1] if n > 1 else [])

    def build_pos(tour):
        pos = np.empty(n, dtype=int)
        for idx, city in enumerate(tour):
            pos[city] = idx
        return pos

    def regret_insertion(start=None):
        if start is None:
            start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            best_costs = []
            for city in unvisited:
                min_cost = float('inf')
                min_pos = -1
                second_min_cost = float('inf')
                for i in range(len(tour)):
                    a = tour[i]
                    b = tour[(i+1) % len(tour)]
                    cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                    if cost < min_cost:
                        second_min_cost = min_cost
                        min_cost = cost
                        min_pos = i+1
                    elif cost < second_min_cost:
                        second_min_cost = cost
                regret = second_min_cost - min_cost
                best_costs.append((city, min_cost, min_pos, regret))
            max_regret = max(c[3] for c in best_costs)
            candidates = [c for c in best_costs if c[3] == max_regret]
            city, cost, pos, _ = candidates[np.random.randint(len(candidates))]
            tour.insert(pos, city)
            unvisited.remove(city)
        return np.array(tour, dtype=np.int32)

    def compute_dist(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))

    tour = regret_insertion()
    pos = build_pos(tour)
    cur_dist = compute_dist(tour)
    best_tour = tour.copy()
    best_dist = cur_dist
    report_best_tour(best_tour)

    iteration = 0
    last_improvement = 0
    restart_threshold = max(10, budget // 30)

    def two_opt_improve():
        nonlocal iteration, cur_dist, best_dist, best_tour, last_improvement, tour, pos
        improved = False
        i = 0
        while i < n - 2 and iteration < budget:
            a = tour[i]
            b = tour[(i+1) % n]
            for c in nn_list[b]:
                j = pos[c]
                if j <= i+1 or j >= n-1:
                    continue
                d = tour[(j+1) % n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < -1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    for idx in range(i+1, j+1):
                        pos[tour[idx]] = idx
                    cur_dist += delta
                    if cur_dist < best_dist - 1e-12:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                        last_improvement = iteration
                    improved = True
                    break
            if improved:
                break
            i += 1
            iteration += 1
        return improved

    def three_opt_search():
        nonlocal iteration, cur_dist, best_dist, best_tour, last_improvement, tour, pos
        n_attempts = min(n, max(1, budget // 10))
        for _ in range(n_attempts):
            if iteration >= budget:
                break
            i = np.random.randint(n-2)
            j = np.random.randint(i+2, n-1)
            k = np.random.randint(j+2, n)
            # four reconnection cases (excluding original)
            # case 1: reverse i+1..j
            # case 2: reverse j+1..k
            # case 3: reverse i+1..j and j+1..k
            # We'll compute delta for each and apply if improvement
            a = tour[i]
            b = tour[(i+1)%n]
            c = tour[j]
            d = tour[(j+1)%n]
            e = tour[k]
            f = tour[(k+1)%n]
            # Precompute original edges
            orig = distance_matrix[a,b] + distance_matrix[c,d] + distance_matrix[e,f]
            # Case 1: reverse segment i+1..j => new edges: a->d, c->b, e->f
            new1 = distance_matrix[a,d] + distance_matrix[c,b] + distance_matrix[e,f]
            delta1 = new1 - orig
            # Case 2: reverse segment j+1..k => new edges: a->b, c->f, e->d
            new2 = distance_matrix[a,b] + distance_matrix[c,f] + distance_matrix[e,d]
            delta2 = new2 - orig
            # Case 3: reverse both segments => new edges: a->d, c->f, e->b
            new3 = distance_matrix[a,d] + distance_matrix[c,f] + distance_matrix[e,b]
            delta3 = new3 - orig
            # Choose best negative delta
            best_delta = 0
            best_case = -1
            if delta1 < best_delta:
                best_delta = delta1
                best_case = 1
            if delta2 < best_delta:
                best_delta = delta2
                best_case = 2
            if delta3 < best_delta:
                best_delta = delta3
                best_case = 3
            if best_case < 0:
                iteration += 1
                continue
            # Apply the best case
            if best_case == 1:
                tour[i+1:j+1] = tour[i+1:j+1][::-1]
            elif best_case == 2:
                tour[j+1:k+1] = tour[j+1:k+1][::-1]
            elif best_case == 3:
                temp = tour[i+1:j+1][::-1].tolist() + tour[j+1:k+1][::-1].tolist()
                tour[i+1:k+1] = temp
            # Update positions
            for idx in range(i+1, k+1):
                pos[tour[idx]] = idx
            cur_dist += best_delta
            if cur_dist < best_dist - 1e-12:
                best_dist = cur_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
                last_improvement = iteration
            iteration += 1
            return True  # applied one move
        return False

    while iteration < budget:
        if not two_opt_improve():
            if iteration - last_improvement > restart_threshold and iteration < budget:
                tour = regret_insertion(start=np.random.randint(n))
                pos = build_pos(tour)
                cur_dist = compute_dist(tour)
                if cur_dist < best_dist - 1e-12:
                    best_dist = cur_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                last_improvement = iteration
            else:
                if iteration < budget:
                    three_opt_search()
                else:
                    break
    return best_tour