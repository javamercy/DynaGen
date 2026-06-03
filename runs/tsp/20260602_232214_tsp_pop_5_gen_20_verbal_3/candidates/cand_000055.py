import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    rng = np.random.default_rng()

    def tour_length(tour):
        return sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))

    def nearest_neighbor(start):
        tour = [start]
        unvisited = set(range(n)) - {start}
        current = start
        while unvisited:
            next_city = min(unvisited, key=lambda c: distance_matrix[current, c])
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        return np.array(tour)

    def cheapest_insertion():
        # start with the edge with largest distance
        i, j = np.unravel_index(np.argmax(distance_matrix), (n, n))
        tour = [i, j]
        unvisited = set(range(n)) - {i, j}
        while unvisited:
            best_cost = float('inf')
            best_city = None
            best_pos = None
            for city in unvisited:
                for k in range(len(tour)):
                    a = tour[k]
                    b = tour[(k+1) % len(tour)]
                    cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                    if cost < best_cost:
                        best_cost = cost
                        best_city = city
                        best_pos = k+1
            tour.insert(best_pos, best_city)
            unvisited.remove(best_city)
        return np.array(tour)

    def regret_insertion():
        i, j = np.unravel_index(np.argmax(distance_matrix), (n, n))
        tour = [i, j]
        unvisited = set(range(n)) - {i, j}
        while unvisited:
            best_cost = {}
            second_cost = {}
            best_pos = {}
            for city in unvisited:
                best = float('inf')
                second = float('inf')
                pos = 0
                for k in range(len(tour)):
                    a = tour[k]
                    b = tour[(k+1) % len(tour)]
                    cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                    if cost < best:
                        second = best
                        best = cost
                        pos = k+1
                    elif cost < second:
                        second = cost
                best_cost[city] = best
                second_cost[city] = second if second != float('inf') else best
                best_pos[city] = pos
            max_regret = max(second_cost[c] - best_cost[c] for c in unvisited)
            candidates = [c for c in unvisited if second_cost[c] - best_cost[c] == max_regret]
            if len(candidates) > 1:
                chosen = min(candidates, key=lambda c: best_cost[c])
            else:
                chosen = candidates[0]
            pos = best_pos[chosen]
            tour.insert(pos, chosen)
            unvisited.remove(chosen)
        return np.array(tour)

    def two_opt(tour):
        best_tour = tour.copy()
        best_dist = tour_length(best_tour)
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a = best_tour[i]
                    b = best_tour[i+1]
                    c = best_tour[j]
                    d = best_tour[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-10:
                        new_tour = best_tour.copy()
                        new_tour[i+1:j+1] = best_tour[j:i:-1]
                        new_dist = best_dist + delta
                        if new_dist < best_dist - 1e-10:
                            best_tour = new_tour
                            best_dist = new_dist
                            improved = True
                            report_best_tour(best_tour)
        return best_tour, best_dist

    # Generate initial tours
    starts = rng.choice(range(n), size=min(3, n), replace=False)
    initial_tours = []
    for start in starts:
        tour = nearest_neighbor(start)
        initial_tours.append(tour)
    initial_tours.append(cheapest_insertion())
    initial_tours.append(regret_insertion())

    best_tour = initial_tours[0].copy()
    best_dist = tour_length(best_tour)
    for tour in initial_tours:
        tour, dist = two_opt(tour)
        if dist < best_dist:
            best_tour = tour
            best_dist = dist
            report_best_tour(best_tour)

    def double_bridge(tour):
        cuts = sorted(rng.choice(np.arange(1, n-1), size=4, replace=False))
        a, b, c, d = cuts[0], cuts[1], cuts[2], cuts[3]
        perturbed = np.concatenate([tour[:a+1], tour[c+1:d+1], tour[b+1:c+1], tour[a+1:b+1], tour[d+1:]])
        if len(perturbed) != n:
            return tour
        return perturbed

    def swap_edges(tour):
        i = rng.integers(0, n-2)
        j = rng.integers(i+2, n)
        new_tour = tour.copy()
        new_tour[i+1:j+1] = tour[j:i:-1]
        return new_tour

    def shift_segment(tour):
        i = rng.integers(0, n-2)
        j = rng.integers(i+2, min(n, i+10))
        k = rng.integers(0, n-1)
        segment = tour[i:j+1]
        rest = np.concatenate([tour[:i], tour[j+1:]])
        new_tour = np.concatenate([rest[:k], segment, rest[k:]])
        return new_tour

    perturbations = [double_bridge, swap_edges, shift_segment]
    max_ils_iters = max(30, int(np.ceil(n / 2)))
    for _ in range(max_ils_iters):
        perturb_f = rng.choice(perturbations)
        perturbed = perturb_f(best_tour)
        new_tour, new_dist = two_opt(perturbed)
        if new_dist < best_dist:
            best_tour = new_tour
            best_dist = new_dist
            report_best_tour(best_tour)

    return best_tour