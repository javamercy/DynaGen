import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=int)
        report_best_tour(tour.copy())
        return tour
    if n == 2:
        tour = np.array([0,1], dtype=int)
        report_best_tour(tour.copy())
        return tour

    def compute_cost(tour):
        return distance_matrix[tour[-1], tour[0]] + np.sum(distance_matrix[tour[:-1], tour[1:]])

    # Nearest neighbor
    start = 0
    tour = [start]
    visited = {start}
    current = start
    while len(tour) < n:
        next_node = min((i for i in range(n) if i not in visited), key=lambda i: distance_matrix[current][i])
        tour.append(next_node)
        visited.add(next_node)
        current = next_node
    tour = np.array(tour, dtype=int)
    best_cost = compute_cost(tour)
    best_tour = tour.copy()
    report_best_tour(best_tour.copy())

    # MST via Prim
    key = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    in_mst = np.zeros(n, dtype=bool)
    key[0] = 0
    for _ in range(n):
        u = np.argmin(key[~in_mst])
        # convert to global index
        u = np.where(~in_mst)[0][u]
        in_mst[u] = True
        for v in range(n):
            if not in_mst[v] and distance_matrix[u][v] < key[v]:
                key[v] = distance_matrix[u][v]
                parent[v] = u
    # Build adjacency
    mst_adj = [[] for _ in range(n)]
    for v in range(1, n):
        u = parent[v]
        mst_adj[u].append(v)
        mst_adj[v].append(u)
    # DFS preorder
    visited = np.zeros(n, dtype=bool)
    preorder = []
    def dfs(u):
        visited[u] = True
        preorder.append(u)
        for v in mst_adj[u]:
            if not visited[v]:
                dfs(v)
    dfs(0)
    tour_mst = np.array(preorder, dtype=int)
    cost_mst = compute_cost(tour_mst)
    if cost_mst < best_cost:
        best_cost = cost_mst
        best_tour = tour_mst.copy()
        report_best_tour(best_tour.copy())

    # 2-opt improvement
    tour = best_tour.copy()
    improved = True
    while improved:
        improved = False
        for i in range(n-1):
            for j in range(i+2, n):
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