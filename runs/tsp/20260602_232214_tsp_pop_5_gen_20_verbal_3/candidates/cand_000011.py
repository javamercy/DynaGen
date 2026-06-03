import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # Prim's algorithm for MST
    in_mst = [False] * n
    key = [float('inf')] * n
    parent = [-1] * n
    key[0] = 0
    for _ in range(n):
        u = min((i for i in range(n) if not in_mst[i]), key=lambda i: key[i])
        in_mst[u] = True
        for v in range(n):
            if not in_mst[v] and distance_matrix[u][v] < key[v]:
                key[v] = distance_matrix[u][v]
                parent[v] = u
    # Build MST edges
    mst_edges = []
    for v in range(1, n):
        mst_edges.append((v, parent[v]))
    # Compute degrees in MST
    deg = np.zeros(n, dtype=int)
    for u, v in mst_edges:
        deg[u] += 1
        deg[v] += 1
    # Odd-degree vertices
    odd_vertices = [i for i in range(n) if deg[i] % 2 == 1]
    # Greedy matching on odd vertices
    matched = [False] * n
    matching_edges = []
    for u in odd_vertices:
        if matched[u]:
            continue
        best = -1
        best_dist = float('inf')
        for v in odd_vertices:
            if not matched[v] and v != u and distance_matrix[u][v] < best_dist:
                best_dist = distance_matrix[u][v]
                best = v
        if best != -1:
            matched[u] = True
            matched[best] = True
            matching_edges.append((u, best))
    # Combine edges
    edges = mst_edges + matching_edges
    # Build adjacency list (multigraph)
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # Hierholzer algorithm for Eulerian tour
    stack = [0]
    circuit = []
    while stack:
        u = stack[-1]
        if adj[u]:
            v = adj[u].pop()
            adj[v].remove(u)
            stack.append(v)
        else:
            circuit.append(stack.pop())
    circuit.reverse()  # circuit is in reverse order
    # Shortcut to Hamiltonian tour
    tour = []
    seen = set()
    for node in circuit:
        if node not in seen:
            seen.add(node)
            tour.append(node)
    tour_arr = np.array(tour)
    report_best_tour(tour_arr)
    return tour_arr