import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0])
        report_best_tour(tour)
        return tour
    if n == 2:
        tour = np.array([0, 1])
        report_best_tour(tour)
        return tour

    def total_dist(t):
        idx = np.array(t)
        return distance_matrix[idx[-1], idx[0]] + np.sum(distance_matrix[idx[:-1], idx[1:]])

    def two_opt_best(tour):
        improved = True
        while improved:
            improved = False
            best_gain = 0.0
            best_i = best_j = None
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == n - 1:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[0]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[0]])
                    else:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[j+1]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[j+1]])
                    if delta > best_gain + 1e-12:
                        best_gain = delta
                        best_i, best_j = i, j
            if best_gain > 1e-12:
                i, j = best_i, best_j
                tour[i+1:j+1] = list(reversed(tour[i+1:j+1]))
                improved = True
        return tour

    # Prim's algorithm for MST
    visited = [False] * n
    parent = [-1] * n
    key = [float('inf')] * n
    key[0] = 0
    for _ in range(n):
        u = min([i for i in range(n) if not visited[i]], key=lambda i: key[i])
        visited[u] = True
        for v in range(n):
            if not visited[v] and distance_matrix[u, v] < key[v]:
                key[v] = distance_matrix[u, v]
                parent[v] = u

    # Build doubled edge count matrix
    rem = np.zeros((n, n), dtype=int)
    for v in range(1, n):
        u = parent[v]
        rem[u][v] += 2
        rem[v][u] += 2

    def euler_circuit(start):
        circuit = []
        stack = [start]
        local_rem = rem.copy()
        while stack:
            v = stack[-1]
            found = False
            for u in range(n):
                if local_rem[v][u] > 0:
                    local_rem[v][u] -= 1
                    local_rem[u][v] -= 1
                    stack.append(u)
                    found = True
                    break
            if not found:
                circuit.append(stack.pop())
        circuit.reverse()
        return circuit

    def shortcut(euler):
        visited_set = set()
        tour = []
        for v in euler:
            if v not in visited_set:
                visited_set.add(v)
                tour.append(v)
        return tour

    best_tour = None
    best_dist = float('inf')
    for restart in range(10):
        start = restart % n
        euler = euler_circuit(start)
        tour = shortcut(euler)
        if len(tour) != n:
            continue
        tour = two_opt_best(tour)
        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = np.array(tour)
            report_best_tour(best_tour)
    return best_tour