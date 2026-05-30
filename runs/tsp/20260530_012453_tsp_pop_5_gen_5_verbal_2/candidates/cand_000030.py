import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def total_distance(tour):
        return sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))

    def farthest_insertion():
        # farthest insertion construction (deterministic start)
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
                min_dist = min(distance_matrix[node][t] for t in tour)
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_node = node
            best_pos = -1
            best_increase = float('inf')
            for i in range(len(tour)):
                a = tour[i]
                b = tour[(i+1)%len(tour)]
                increase = distance_matrix[a][best_node] + distance_matrix[best_node][b] - distance_matrix[a][b]
                if increase < best_increase:
                    best_increase = increase
                    best_pos = i+1
            tour.insert(best_pos, best_node)
            in_tour.add(best_node)
        return tour

    def steepest_2opt(tour):
        improved = True
        best_tour = tour.copy()
        best_cost = total_distance(tour)
        while improved:
            improved = False
            best_delta = 0
            best_i = best_j = -1
            for i in range(n):
                for j in range(i+2, n):
                    if i == 0 and j == n-1:
                        continue
                    a = tour[i]
                    b = tour[(i+1)%n]
                    c = tour[j]
                    d = tour[(j+1)%n]
                    delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            if best_delta < 0:
                i, j = best_i, best_j
                tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                new_cost = total_distance(tour)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_tour = tour.copy()
                    report_best_tour(np.array(best_tour))
                improved = True
        return best_tour, best_cost

    # First construction and steepest descent
    tour = farthest_insertion()
    best_tour, best_cost = steepest_2opt(tour)
    report_best_tour(np.array(best_tour))

    # Multi-start with random perturbations
    max_restarts = 5
    num_perturb = 5
    for _ in range(max_restarts):
        # Perturb current best tour with random 2-opt moves
        perturbed = best_tour.copy()
        for _ in range(num_perturb):
            i = np.random.randint(0, n-2)
            j = np.random.randint(i+2, n)
            if i == 0 and j == n-1:
                continue
            perturbed = perturbed[:i+1] + perturbed[i+1:j+1][::-1] + perturbed[j+1:]
        # Apply steepest 2-opt
        new_tour, new_cost = steepest_2opt(perturbed)
        if new_cost < best_cost:
            best_cost = new_cost
            best_tour = new_tour
            report_best_tour(np.array(best_tour))
    return np.array(best_tour)