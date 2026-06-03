import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=int)
        report_best_tour(tour.copy())
        return tour
    if n == 2:
        tour = np.array([0, 1], dtype=int)
        report_best_tour(tour.copy())
        return tour

    # Prim's MST
    INF = 1e100
    visited = [False] * n
    parent = [-1] * n
    key = [INF] * n
    key[0] = 0
    for _ in range(n):
        u = min(range(n), key=lambda i: key[i] if not visited[i] else INF)
        visited[u] = True
        for v in range(n):
            if not visited[v] and distance_matrix[u][v] < key[v]:
                key[v] = distance_matrix[u][v]
                parent[v] = u

    # Build adjacency list of MST
    mst_adj = [[] for _ in range(n)]
    for v in range(1, n):
        u = parent[v]
        mst_adj[u].append(v)
        mst_adj[v].append(u)

    # DFS preorder to get initial tour
    tour = []
    stack = [0]
    visited_order = [False] * n
    while stack:
        u = stack.pop()
        if not visited_order[u]:
            visited_order[u] = True
            tour.append(u)
            for v in mst_adj[u]:
                if not visited_order[v]:
                    stack.append(v)
    # tour is now a permutation of 0..n-1
    tour = np.array(tour, dtype=int)

    def compute_cost(t):
        return distance_matrix[t[-1], t[0]] + np.sum(distance_matrix[t[:-1], t[1:]])

    best_cost = compute_cost(tour)
    best_tour = tour.copy()
    report_best_tour(best_tour.copy())

    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                new_tour = tour.copy()
                new_tour[i+1:j+1] = new_tour[i+1:j+1][::-1]
                new_cost = compute_cost(new_tour)
                if new_cost < best_cost - 1e-10:
                    best_cost = new_cost
                    best_tour = new_tour.copy()
                    tour = new_tour.copy()
                    improved = True
                    report_best_tour(best_tour.copy())
                    break
            if improved:
                break

    return best_tour