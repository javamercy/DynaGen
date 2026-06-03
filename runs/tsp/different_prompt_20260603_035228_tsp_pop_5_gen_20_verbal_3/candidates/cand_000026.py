import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    # Prim's algorithm to compute MST adjacency list
    visited = [False] * n
    min_edge = [float('inf')] * n
    parent = [-1] * n
    min_edge[0] = 0
    for _ in range(n):
        u = min((val, i) for i, val in enumerate(min_edge) if not visited[i])[1]
        visited[u] = True
        for v in range(n):
            if not visited[v] and distance_matrix[u][v] < min_edge[v]:
                min_edge[v] = distance_matrix[u][v]
                parent[v] = u
    adj = [[] for _ in range(n)]
    for v in range(1, n):
        u = parent[v]
        adj[u].append(v)
        adj[v].append(u)

    # DFS to get preorder tour
    def dfs(u, visited, order):
        visited[u] = True
        order.append(u)
        for v in adj[u]:
            if not visited[v]:
                dfs(v, visited, order)
    visited = [False] * n
    tour = []
    dfs(0, visited, tour)

    # Compute total distance
    total = distance_matrix[tour[-1], tour[0]]
    for i in range(n - 1):
        total += distance_matrix[tour[i], tour[i+1]]
    best_dist = total
    best_tour = tour[:]
    report_best_tour(np.array(best_tour))

    # 2-opt first-improvement local search
    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                a = tour[i]
                b = tour[i+1]
                c = tour[j]
                d = tour[(j+1) % n]
                delta = distance_matrix[a][b] + distance_matrix[c][d] - distance_matrix[a][c] - distance_matrix[b][d]
                if delta > 0:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    total -= delta
                    improved = True
                    if total < best_dist:
                        best_dist = total
                        best_tour = tour[:]
                        report_best_tour(np.array(best_tour))
                    break
            if improved:
                break
    return np.array(best_tour)