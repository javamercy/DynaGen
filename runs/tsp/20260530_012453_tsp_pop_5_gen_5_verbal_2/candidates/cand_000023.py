import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def total_distance(tour):
        return distance_matrix[tour, np.roll(tour, -1)].sum()

    # Farthest insertion construction
    start = 0
    end = np.argmax(distance_matrix[start])
    tour = [start, end]
    in_tour = {start, end}
    while len(tour) < n:
        best_node = -1
        best_dist = -1.0
        for node in range(n):
            if node in in_tour:
                continue
            min_dist = min(distance_matrix[node, t] for t in tour)
            if min_dist > best_dist:
                best_dist = min_dist
                best_node = node
        best_pos = -1
        best_increase = float('inf')
        for i in range(len(tour)):
            a = tour[i]
            b = tour[(i+1)%len(tour)]
            increase = distance_matrix[a, best_node] + distance_matrix[best_node, b] - distance_matrix[a, b]
            if increase < best_increase:
                best_increase = increase
                best_pos = i+1
        tour.insert(best_pos, best_node)
        in_tour.add(best_node)
    best_tour = tour[:]
    best_cost = total_distance(best_tour)
    report_best_tour(np.array(best_tour))

    # First-improving 2-opt local search
    def first_improving_2opt(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    if i == 0 and j == n-1:
                        continue
                    a, b = tour[i], tour[(i+1)%n]
                    c, d = tour[j], tour[(j+1)%n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-12:
                        tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                        improved = True
                        break
                if improved:
                    break
        return tour

    tour = first_improving_2opt(tour)
    cur_cost = total_distance(tour)
    if cur_cost < best_cost - 1e-12:
        best_cost = cur_cost
        best_tour = tour[:]
        report_best_tour(np.array(best_tour))

    # Iterated local search with random perturbations
    max_iters = 5
    for _ in range(max_iters):
        # random 2-opt perturbation (reverse a random segment)
        i = np.random.randint(0, n-2)
        j = np.random.randint(i+2, n)
        if i == 0 and j == n-1:
            continue
        perturbed = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
        perturbed = first_improving_2opt(perturbed)
        new_cost = total_distance(perturbed)
        if new_cost < cur_cost - 1e-12:
            cur_cost = new_cost
            tour = perturbed[:]
            if new_cost < best_cost - 1e-12:
                best_cost = new_cost
                best_tour = perturbed[:]
                report_best_tour(np.array(best_tour))
        # else: keep current tour

    return np.array(best_tour)