import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def total_distance(tour):
        return sum(distance_matrix[tour[i]][tour[(i+1)%n]] for i in range(n))

    def farthest_insertion():
        start = np.random.randint(n)
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
        best = tour[:]
        best_cost = total_distance(best)
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
                    delta = (distance_matrix[a][c] + distance_matrix[b][d]) - (distance_matrix[a][b] + distance_matrix[c][d])
                    if delta < best_delta and delta < 0:
                        best_delta = delta
                        best_i, best_j = i, j
            if best_i != -1:
                i, j = best_i, best_j
                tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                new_cost = total_distance(tour)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best = tour[:]
                    report_best_tour(np.array(best))
                improved = True
        return best, best_cost

    def double_bridge(tour):
        n_tour = len(tour)
        while True:
            a = np.random.randint(0, n_tour // 3)
            b = np.random.randint(a + 1, n_tour // 2)
            c = np.random.randint(b + 1, 2 * n_tour // 3)
            d = np.random.randint(c + 1, n_tour - 1)
            if a < b < c < d:
                break
        seg1 = tour[:a]
        seg2 = tour[a:b]
        seg3 = tour[b:c]
        seg4 = tour[c:d]
        seg5 = tour[d:]
        new_tour = seg1 + seg3 + seg2 + seg5 + seg4
        return new_tour

    # Initial construction
    initial_tour = farthest_insertion()
    best_tour, best_cost = steepest_2opt(initial_tour)
    report_best_tour(np.array(best_tour))

    # Iterated local search
    no_improve_counter = 0
    max_no_improve = 5
    while no_improve_counter < max_no_improve:
        # Perturbation: double-bridge
        perturbed = double_bridge(best_tour)
        # Local search
        new_tour, new_cost = steepest_2opt(perturbed)
        if new_cost < best_cost:
            best_cost = new_cost
            best_tour = new_tour[:]
            report_best_tour(np.array(best_tour))
            no_improve_counter = 0
        else:
            no_improve_counter += 1
        # Restart if stuck
        if no_improve_counter >= max_no_improve:
            # Random restart with farthest insertion from a random start
            initial_tour = farthest_insertion()
            # Instead of full steepest_2opt, use a quick 2-opt? For compactness, just replace best_tour with a random permutation
            # But to maintain quality, we re-run steepest_2opt on a new farthest insertion
            new_tour, new_cost = steepest_2opt(initial_tour)
            if new_cost < best_cost:
                best_cost = new_cost
                best_tour = new_tour[:]
                report_best_tour(np.array(best_tour))
            no_improve_counter = 0

    return np.array(best_tour)