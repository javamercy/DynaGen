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
    improvements_since_last_restart = 0
    restart_threshold = max(10, budget // 30)
    consecutive_no_improve = 0

    def two_opt_improve():
        nonlocal iteration, cur_dist, best_dist, best_tour, last_improvement, tour, pos, consecutive_no_improve, improvements_since_last_restart
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
                        improvements_since_last_restart += 1
                    consecutive_no_improve = 0
                    improved = True
                    break
            if improved:
                break
            i += 1
            iteration += 1
        if not improved:
            consecutive_no_improve += 1
        return improved

    def three_opt_search():
        nonlocal iteration, cur_dist, best_dist, best_tour, last_improvement, tour, pos, consecutive_no_improve, improvements_since_last_restart
        n_attempts = min(n, max(1, budget // 10))
        attempts = 0
        while attempts < n_attempts and iteration < budget:
            i = np.random.randint(n-2)
            j = np.random.randint(i+2, n-1)
            k = np.random.randint(j+2, n)
            a = tour[i]
            b = tour[(i+1)%n]
            c = tour[j]
            d = tour[(j+1)%n]
            e = tour[k]
            f = tour[(k+1)%n]
            orig = distance_matrix[a,b] + distance_matrix[c,d] + distance_matrix[e,f]
            # Case 1: reverse i+1..j
            new1 = distance_matrix[a,d] + distance_matrix[c,b] + distance_matrix[e,f]
            delta1 = new1 - orig
            # Case 2: reverse j+1..k
            new2 = distance_matrix[a,b] + distance_matrix[c,f] + distance_matrix[e,d]
            delta2 = new2 - orig
            # Case 3: reverse both
            new3 = distance_matrix[a,d] + distance_matrix[c,f] + distance_matrix[e,b]
            delta3 = new3 - orig
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
                attempts += 1
                iteration += 1
                continue
            if best_case == 1:
                tour[i+1:j+1] = tour[i+1:j+1][::-1]
            elif best_case == 2:
                tour[j+1:k+1] = tour[j+1:k+1][::-1]
            else:
                temp = tour[i+1:j+1][::-1].tolist() + tour[j+1:k+1][::-1].tolist()
                tour[i+1:k+1] = temp
            for idx in range(i+1, k+1):
                pos[tour[idx]] = idx
            cur_dist += best_delta
            if cur_dist < best_dist - 1e-12:
                best_dist = cur_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
                last_improvement = iteration
                improvements_since_last_restart += 1
            consecutive_no_improve = 0
            iteration += 1
            attempts += 1
            return True  # applied one move, will continue in outer loop
        return False

    while iteration < budget:
        if not two_opt_improve():
            if iteration - last_improvement > restart_threshold and iteration < budget:
                # dynamic adjustment: increase threshold if recent improvements
                if improvements_since_last_restart > 0:
                    restart_threshold = min(restart_threshold * 2, budget // 10)
                else:
                    restart_threshold = max(restart_threshold // 2, 5)
                improvements_since_last_restart = 0
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
        else:
            # after 2-opt improvement, update restart threshold dynamically
            if improvements_since_last_restart > 5:
                restart_threshold = max(restart_threshold // 2, 5)
    return best_tour