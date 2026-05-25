import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        np.random.default_rng(seed).shuffle(tour)
        return tour
    rng = np.random.default_rng(seed)
    dist = distance_matrix

    def regret_construction(start):
        tour = [start]
        unvisited = set(range(n)) - {start}
        while unvisited:
            best_data = {}
            for city in unvisited:
                best_cost = np.inf
                second_best = np.inf
                best_pos = -1
                m = len(tour)
                for i in range(m):
                    prev = tour[i]
                    nxt = tour[(i+1) % m]
                    inc = dist[prev, city] + dist[city, nxt] - dist[prev, nxt]
                    if inc < best_cost:
                        second_best = best_cost
                        best_cost = inc
                        best_pos = i+1
                    elif inc < second_best:
                        second_best = inc
                regret = second_best - best_cost if second_best != np.inf else best_cost
                best_data[city] = (regret, best_cost, best_pos)
            max_regret = max(v[0] for v in best_data.values())
            candidates = [c for c, v in best_data.items() if v[0] == max_regret]
            chosen = candidates[rng.integers(len(candidates))]
            pos = best_data[chosen][2]
            tour.insert(pos, chosen)
            unvisited.remove(chosen)
        return np.array(tour, dtype=np.int64)

    def random_insertion():
        tour = [rng.integers(n)]
        unvisited = set(range(n)) - {tour[0]}
        for city in rng.permutation(list(unvisited)):
            best_cost = np.inf
            best_pos = -1
            m = len(tour)
            for i in range(m):
                prev = tour[i]
                nxt = tour[(i+1) % m]
                inc = dist[prev, city] + dist[city, nxt] - dist[prev, nxt]
                if inc < best_cost:
                    best_cost = inc
                    best_pos = i+1
            tour.insert(best_pos, city)
        return np.array(tour, dtype=np.int64)

    start = rng.integers(n)
    tour = regret_construction(start)
    best_tour = tour.copy()
    best_dist = 0.0
    for i in range(n):
        best_dist += dist[tour[i], tour[(i+1)%n]]
    report_best_tour(best_tour)
    cur_tour = tour.copy()
    cur_dist = best_dist
    ops = 0
    no_improve_passes = 0
    threshold = max(2, n // 8)  # lower threshold for earlier diversification
    temperature = 0.1  # initial temperature for acceptance
    cooling = 0.999

    while ops < budget:
        # First-improvement 2-opt pass
        improved = False
        for i in range(n-1):
            if ops >= budget:
                break
            for j in range(i+2, n):
                if ops >= budget:
                    break
                ops += 1
                a, b = cur_tour[i], cur_tour[(i+1)%n]
                c, d = cur_tour[j], cur_tour[(j+1)%n]
                old = dist[a,b] + dist[c,d]
                new = dist[a,c] + dist[b,d]
                if new < old - 1e-12:
                    cur_tour[i+1:j+1] = cur_tour[i+1:j+1][::-1]
                    cur_dist += new - old
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = cur_tour.copy()
                        report_best_tour(best_tour)
                    improved = True
                    break
            if improved:
                break
        if ops >= budget:
            break
        if improved:
            no_improve_passes = 0
            # Occasionally perturb even after improvement
            if rng.random() < 0.1:
                if ops >= budget:
                    break
                ops += 1
                # small perturbation: random swap of two cities
                i, j = rng.choice(n, size=2, replace=False)
                cur_tour[i], cur_tour[j] = cur_tour[j], cur_tour[i]
                cur_dist = 0.0
                for k in range(n):
                    cur_dist += dist[cur_tour[k], cur_tour[(k+1)%n]]
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_tour = cur_tour.copy()
                    report_best_tour(best_tour)
        else:
            no_improve_passes += 1
            # Trigger diversification more often
            if no_improve_passes >= threshold or rng.random() < 0.2:
                if ops >= budget:
                    break
                ops += 1
                if rng.random() < 0.5:
                    # Large perturbation: remove random segment and reinsert greedily
                    L = max(2, n // 4)
                    start_idx = rng.integers(0, n - L + 1)
                    segment = set(cur_tour[start_idx:start_idx+L])
                    base = [c for c in cur_tour if c not in segment]
                    for city in segment:
                        best_cost = np.inf
                        best_pos = -1
                        m = len(base)
                        for i in range(m):
                            prev = base[i]
                            nxt = base[(i+1)%m]
                            inc = dist[prev, city] + dist[city, nxt] - dist[prev, nxt]
                            if inc < best_cost:
                                best_cost = inc
                                best_pos = i+1
                        base.insert(best_pos, city)
                    new_tour = np.array(base, dtype=np.int64)
                else:
                    # Random 2-opt move
                    i = rng.integers(0, n-1)
                    j = rng.integers(i+2, n)
                    new_tour = cur_tour.copy()
                    new_tour[i+1:j+1] = new_tour[i+1:j+1][::-1]
                new_dist = 0.0
                for k in range(n):
                    new_dist += dist[new_tour[k], new_tour[(k+1)%n]]
                # Update best
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_tour = new_tour.copy()
                    report_best_tour(best_tour)
                # Accept with simulated annealing probability
                delta = new_dist - cur_dist
                if delta < 0:
                    accept = True
                else:
                    accept = rng.random() < np.exp(-delta / (temperature * best_dist + 1e-8))
                if accept:
                    cur_tour = new_tour
                    cur_dist = new_dist
                # Cool down
                temperature *= cooling
                no_improve_passes = 0
            # Restart from new construction if stagnation persists
            if no_improve_passes >= 2 * threshold:
                if ops >= budget:
                    break
                if rng.random() < 0.5:
                    cur_tour = random_insertion()
                else:
                    new_start = rng.integers(n)
                    cur_tour = regret_construction(new_start)
                cur_dist = 0.0
                for i in range(n):
                    cur_dist += dist[cur_tour[i], cur_tour[(i+1)%n]]
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_tour = cur_tour.copy()
                    report_best_tour(best_tour)
                no_improve_passes = 0
                temperature = 0.1  # reset temperature on restart
    return best_tour