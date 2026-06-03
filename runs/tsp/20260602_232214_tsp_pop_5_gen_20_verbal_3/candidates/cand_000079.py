import numpy as np
def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # Prim's MST
    in_mst = [False]*n
    key = [float('inf')]*n
    parent = [-1]*n
    key[0]=0
    for _ in range(n):
        u = min((i for i in range(n) if not in_mst[i]), key=lambda i: key[i])
        in_mst[u]=True
        for v in range(n):
            if not in_mst[v] and distance_matrix[u][v] < key[v]:
                key[v]=distance_matrix[u][v]; parent[v]=u
    mst_edges = [(v,parent[v]) for v in range(1,n)]
    deg = np.zeros(n, int)
    for u,v in mst_edges:
        deg[u]+=1; deg[v]+=1
    odd = [i for i in range(n) if deg[i]%2==1]
    # Greedy matching
    matched = [False]*n
    matching_edges = []
    for u in odd:
        if matched[u]: continue
        best=-1; best_dist=float('inf')
        for v in odd:
            if not matched[v] and v!=u and distance_matrix[u][v] < best_dist:
                best_dist=distance_matrix[u][v]; best=v
        if best!=-1:
            matched[u]=True; matched[best]=True; matching_edges.append((u,best))
    edges = mst_edges + matching_edges
    adj = {i:[] for i in range(n)}
    for u,v in edges:
        adj[u].append(v); adj[v].append(u)
    # Hierholzer
    stack=[0]; circuit=[]
    while stack:
        u=stack[-1]
        if adj[u]:
            v=adj[u].pop()
            adj[v].remove(u)
            stack.append(v)
        else:
            circuit.append(stack.pop())
    circuit.reverse()
    # Shortcut
    tour=[]
    seen=set()
    for node in circuit:
        if node not in seen:
            seen.add(node); tour.append(node)
    tour = np.array(tour)
    best_tour = tour.copy()
    best_dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour)
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n-1):
            for j in range(i+2, n):
                if j-i==1: continue
                a,b,c,d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                old = distance_matrix[a,b] + distance_matrix[c,d]
                new = distance_matrix[a,c] + distance_matrix[b,d]
                if new < old:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
        if improved:
            new_dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
            if new_dist < best_dist:
                best_dist = new_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
    return best_tour