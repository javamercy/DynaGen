import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)

    k = min(n-1, 30)
    nn = [np.argsort(distance_matrix[i])[1:k+1] for i in range(n)]

    def build_pos(tour):
        pos = np.empty(n, dtype=int)
        for idx, city in enumerate(tour):
            pos[city] = idx
        return pos

    def regret_insertion():
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            best = []
            for city in unvisited:
                best_cost = float('inf')
                best_pos = -1
                second = float('inf')
                for i in range(len(tour)):
                    a = tour[i]
                    b = tour[(i+1) % len(tour)]
                    cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                    if cost < best_cost:
                        second = best_cost
                        best_cost = cost
                        best_pos = i+1
                    elif cost < second:
                        second = cost
                regret = second - best_cost
                best.append((city, best_cost, best_pos, regret))
            max_r = max(b[3] for b in best)
            cand = [b for b in best if b[3] == max_r]
            city, _, pos, _ = cand[np.random.randint(len(cand))]
            tour.insert(pos, city)
            unvisited.remove(city)
        return np.array(tour, dtype=np.int32)

    tour = regret_insertion()
    pos = build_pos(tour)
    best_tour = tour.copy()
    best_dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour)
    curr_dist = best_dist
    iteration = 0
    last_improvement = 0
    restart_threshold = max(10, budget // 15)
    improved_this_phase = True

    while iteration < budget:
        if improved_this_phase:
            best_move = None
            best_delta = 0.0
            for i in range(n-2):
                a = tour[i]
                b = tour[i+1]
                for c in nn[b]:
                    j = pos[c]
                    if j <= i+1 or j >= n-1:
                        continue
                    d = tour[(j+1) % n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < best_delta - 1e-12:
                        best_delta = delta
                        best_move = (i, j)
                    iteration += 1
                    if iteration >= budget:
                        break
                if iteration >= budget:
                    break
            if best_move is not None and best_delta < -1e-12:
                i, j = best_move
                tour[i+1:j+1] = tour[i+1:j+1][::-1]
                for idx in range(i+1, j+1):
                    pos[tour[idx]] = idx
                curr_dist += best_delta
                if curr_dist < best_dist - 1e-12:
                    best_dist = curr_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                    last_improvement = iteration
                improved_this_phase = True
            else:
                improved_this_phase = False

        # Stagnation check and perturbation
        if iteration - last_improvement > restart_threshold and iteration < budget:
            i = np.random.randint(0, n-2)
            b = tour[i+1]
            candidates = nn[b]
            if len(candidates) > 0:
                j = np.random.choice(candidates)
                if j <= i+1 or j >= n-1:
                    j = np.random.randint(i+2, n-1)
            else:
                j = np.random.randint(i+2, n-1)
            delta = distance_matrix[tour[i], tour[j]] + distance_matrix[tour[i+1], tour[(j+1)%n]] - distance_matrix[tour[i], tour[i+1]] - distance_matrix[tour[j], tour[(j+1)%n]]
            tour[i+1:j+1] = tour[i+1:j+1][::-1]
            for idx in range(i+1, j+1):
                pos[tour[idx]] = idx
            curr_dist += delta
            if curr_dist < best_dist - 1e-12:
                best_dist = curr_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
            last_improvement = iteration
            improved_this_phase = True
            iteration += 1

        if iteration >= budget:
            break
        # If no improvement and no perturbation needed, continue to next iteration
        if not improved_this_phase and iteration - last_improvement <= restart_threshold:
            # waste a small budget to avoid busy loop? but we need to exit
            break
    return best_tour