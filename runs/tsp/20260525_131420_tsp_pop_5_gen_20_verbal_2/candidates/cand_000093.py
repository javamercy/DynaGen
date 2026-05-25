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

    def tour_length(t):
        return sum(distance_matrix[t[i], t[(i+1) % n]] for i in range(n))

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
        return tour

    # Multiple restarts for initial tour
    best_initial = None
    best_initial_dist = float('inf')
    for _ in range(5):
        tour = regret_insertion()
        dist = tour_length(tour)
        if dist < best_initial_dist:
            best_initial_dist = dist
            best_initial = tour[:]
    tour = best_initial
    pos = build_pos(tour)
    best_tour = tour[:]
    best_dist = best_initial_dist
    report_best_tour(np.array(best_tour))
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
                    best_tour = tour[:]
                    report_best_tour(np.array(best_tour))
                    last_improvement = iteration
                improved_this_phase = True
            else:
                improved_this_phase = False

        # Stagnation check and double-bridge perturbation
        if iteration - last_improvement > restart_threshold and iteration < budget:
            # Generate random valid indices for double-bridge
            attempts = 0
            while attempts < 10:
                a = np.random.randint(0, n-4)
                b = np.random.randint(a+2, n-3)
                c = np.random.randint(b+2, n-2)
                d = np.random.randint(c+2, n-1)  # ensure d+1 < n
                if a < b < c < d and d+1 < n:
                    break
                attempts += 1
            if attempts == 10:
                # fallback to simple 2-opt swap
                i = np.random.randint(0, n-2)
                j = np.random.randint(i+2, n-1)
                delta = distance_matrix[tour[i], tour[j]] + distance_matrix[tour[i+1], tour[(j+1)%n]] - distance_matrix[tour[i], tour[i+1]] - distance_matrix[tour[j], tour[(j+1)%n]]
                tour[i+1:j+1] = tour[i+1:j+1][::-1]
                for idx in range(i+1, j+1):
                    pos[tour[idx]] = idx
                curr_dist += delta
            else:
                # double-bridge
                old_edges = [distance_matrix[tour[a], tour[a+1]], distance_matrix[tour[b], tour[b+1]], distance_matrix[tour[c], tour[c+1]], distance_matrix[tour[d], tour[(d+1)%n]]]
                new_edges = [distance_matrix[tour[a], tour[c+1]], distance_matrix[tour[c], tour[a+1]], distance_matrix[tour[b], tour[d+1]], distance_matrix[tour[d], tour[b+1]]]
                delta = sum(new_edges) - sum(old_edges)
                # Build new tour
                tour = tour[:a+1] + tour[c+1:d+1] + tour[b+1:c+1] + tour[a+1:b+1] + tour[d+1:]
                # Update position
                for idx, city in enumerate(tour):
                    pos[city] = idx
                curr_dist += delta
            iteration += 1
            if curr_dist < best_dist - 1e-12:
                best_dist = curr_dist
                best_tour = tour[:]
                report_best_tour(np.array(best_tour))
                last_improvement = iteration
            improved_this_phase = True

        if iteration >= budget:
            break
        # If no improvement and no perturbation needed, break to avoid infinite loop
        if not improved_this_phase and iteration - last_improvement <= restart_threshold:
            break
    return np.array(best_tour)