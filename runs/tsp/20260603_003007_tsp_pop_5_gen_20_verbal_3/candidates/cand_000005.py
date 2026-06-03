import numpy as np

def solve_tsp(dist):
    n = len(dist)
    if n <= 3:
        tour = list(range(n))
        report_best_tour(np.array(tour))
        return np.array(tour)
    # Prim's MST
    inMST = [False] * n
    key = [float('inf')] * n
    parent = [-1] * n
    key[0] = 0
    for _ in range(n):
        u = min((i for i in range(n) if not inMST[i]), key=lambda i: key[i])
        inMST[u] = True
        for v in range(n):
            if not inMST[v] and dist[u][v] < key[v]:
                key[v] = dist[u][v]
                parent[v] = u
    mst_edges = []
    for v in range(1, n):
        if parent[v] != -1:
            mst_edges.append((parent[v], v))
    # Build multigraph adjacency for Eulerian circuit
    arcs = []  # list of (from, to)
    adj = [[] for _ in range(n)]
    for u, v in mst_edges:
        for _ in range(2):  # two copies
            arcs.append((u, v))
            arcs.append((v, u))
            adj[u].append(len(arcs)-2)
            adj[v].append(len(arcs)-1)
    used = [False] * len(arcs)
    # Hierholzer
    stack = [0]
    circuit = []
    while stack:
        u = stack[-1]
        # find unused arc
        found = False
        for idx in adj[u]:
            if not used[idx]:
                used[idx] = True
                stack.append(arcs[idx][1])
                found = True
                break
        if not found:
            stack.pop()
            circuit.append(u)
    # Shortcut
    seen = set()
    tour = []
    for node in circuit:
        if node not in seen:
            seen.add(node)
            tour.append(node)
    # Ensure all nodes
    for i in range(n):
        if i not in seen:
            tour.append(i)
    best_tour = tour[:]
    best_dist = sum(dist[tour[i]][tour[(i+1)%n]] for i in range(n))
    report_best_tour(np.array(best_tour))
    # 2-opt
    improved = True
    while improved:
        improved = False
        for i in range(n-2):
            for j in range(i+2, n):
                if j - i == n-1: continue
                a = best_tour[i]; b = best_tour[(i+1) % n]
                c = best_tour[j]; d = best_tour[(j+1) % n]
                delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
                if delta < -1e-12:
                    best_tour[i+1:j+1] = reversed(best_tour[i+1:j+1])
                    best_dist += delta
                    improved = True
                    report_best_tour(np.array(best_tour))
    return np.array(best_tour)