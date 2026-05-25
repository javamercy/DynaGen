import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour)
        return tour

    np.random.seed(seed)

    # candidate lists (first min(15, n-1) nearest neighbors)
    cand_size = min(15, n-1) if n >= 80 else n-1
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        if cand_size < n - 1:
            idx = np.argpartition(dists, cand_size)[:cand_size]
            idx_sorted = idx[np.argsort(dists[idx])]
        else:
            idx_sorted = np.argsort(dists)[1:]
        candidates.append(idx_sorted)

    def tour_length(t):
        total = 0.0
        for i in range(n):
            total += distance_matrix[t[i], t[(i+1)%n]]
        return total

    def regret2_insertion():
        # start with two random cities
        start = np.random.randint(n)
        second = np.random.choice([c for c in range(n) if c != start])
        tour = [start, second]
        unvisited = set(range(n)) - {start, second}
        # for each unvisited city, keep best and second-best insertion cost
        best_cost = {}
        second_best_cost = {}
        best_edge = {}  # stores the best edge index (position of insertion)
        # compute initial insertion costs
        for city in unvisited:
            costs = []
            for i in range(len(tour)):
                j = (i+1) % len(tour)
                cost = distance_matrix[tour[i], city] + distance_matrix[city, tour[j]] - distance_matrix[tour[i], tour[j]]
                costs.append((cost, i+1))
            costs.sort(key=lambda x: x[0])
            best_cost[city] = costs[0][0]
            if len(costs) > 1:
                second_best_cost[city] = costs[1][0]
            else:
                second_best_cost[city] = float('inf')
            best_edge[city] = costs[0][1]

        while unvisited:
            # choose city with maximum regret (difference between best and second best)
            max_regret = -1
            chosen_city = None
            for city in unvisited:
                regret = second_best_cost[city] - best_cost[city]
                if regret > max_regret:
                    max_regret = regret
                    chosen_city = city
            # insert chosen city at its best position
            pos = best_edge[chosen_city]
            tour.insert(pos, chosen_city)
            unvisited.remove(chosen_city)
            # update costs for remaining cities
            # only need to recompute for edges adjacent to the new insertion
            # but for simplicity, recompute all (n is small enough)
            for city in unvisited:
                costs = []
                for i in range(len(tour)):
                    j = (i+1) % len(tour)
                    cost = distance_matrix[tour[i], city] + distance_matrix[city, tour[j]] - distance_matrix[tour[i], tour[j]]
                    costs.append((cost, i+1))
                costs.sort(key=lambda x: x[0])
                best_cost[city] = costs[0][0]
                if len(costs) > 1:
                    second_best_cost[city] = costs[1][0]
                else:
                    second_best_cost[city] = float('inf')
                best_edge[city] = costs[0][1]
        return np.array(tour, dtype=np.int32)

    # initial tour
    tour = regret2_insertion()
    report_best_tour(tour.copy())
    best_tour = tour.copy()
    best_len = tour_length(best_tour)
    eps = 1e-12

    while budget > 0:
        improved = False
        # first-improvement 2-opt over candidate lists
        for i in range(n):
            if budget <= 0:
                break
            ip1 = (i + 1) % n
            a = tour[i]
            b = tour[ip1]
            for j_cand in candidates[b]:
                if budget <= 0:
                    break
                # find position of j_cand in tour
                j = -1
                for idx in range(n):
                    if tour[idx] == j_cand:
                        j = idx
                        break
                if j == -1:
                    continue
                if j == i or j == ip1 or j == (i - 1) % n:
                    continue
                jp1 = (j + 1) % n
                c = tour[j]
                d = tour[jp1]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                budget -= 1
                if delta < -eps:
                    # apply 2-opt reverse
                    if ip1 <= j:
                        tour[ip1:j+1] = tour[ip1:j+1][::-1]
                    else:
                        segment = np.concatenate([tour[ip1:], tour[:j+1]])
                        segment = segment[::-1]
                        tour[ip1:] = segment[:n-ip1]
                        tour[:j+1] = segment[n-ip1:]
                    improved = True
                    new_len = tour_length(tour)
                    if new_len < best_len - eps:
                        best_len = new_len
                        best_tour = tour.copy()
                        report_best_tour(best_tour.copy())
                    break  # first improvement, exit candidate loop
        if not improved and budget > 0:
            # restart with new regret2 insertion tour
            budget -= 1
            tour = regret2_insertion()
            new_len = tour_length(tour)
            if new_len < best_len - eps:
                best_len = new_len
                best_tour = tour.copy()
                report_best_tour(best_tour.copy())
        elif not improved:
            break
    return best_tour