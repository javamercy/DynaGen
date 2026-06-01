import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    m = n - 1
    customers = list(range(1, n))
    
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    
    def route_dist(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def compute_max(routes):
        return max(route_dist(r) for r in routes)
    
    def dp_split(perm):
        # perm: list of customer indices in order
        # Compute segment distances
        seg = [[0]*(m+1) for _ in range(m)]
        for l in range(m):
            acc = distance_matrix[0][perm[l]]
            for r in range(l+1, m+1):
                if r > l+1:
                    acc += distance_matrix[perm[r-2]][perm[r-1]]
                if r == l+1:
                    rd = distance_matrix[0][perm[l]] + distance_matrix[perm[l]][0]
                else:
                    rd = acc + distance_matrix[perm[r-1]][0]
                seg[l][r] = rd
        # DP minimax
        INF = float('inf')
        dp = [[INF]*(truck_count+1) for _ in range(m+1)]
        ch = [[-1]*(truck_count+1) for _ in range(m+1)]
        dp[0][0] = 0
        for i in range(1, m+1):
            for k in range(1, min(i, truck_count)+1):
                best = INF
                best_j = -1
                for j in range(k-1, i):
                    if dp[j][k-1] < INF:
                        cand = max(dp[j][k-1], seg[j][i])
                        if cand < best:
                            best = cand
                            best_j = j
                dp[i][k] = best
                ch[i][k] = best_j
        # reconstruct
        routes = []
        i = m
        k = truck_count
        while k > 0:
            j = ch[i][k]
            seg_indices = perm[j:i]
            routes.append([0] + seg_indices + [0])
            i = j
            k -= 1
        routes.reverse()
        while len(routes) < truck_count:
            routes.append([0,0])
        return routes
    
    # initial random permutation and split
    perm = customers[:]
    random.shuffle(perm)
    routes = dp_split(perm)
    best_routes = [list(r) for r in routes]
    current_max = compute_max(routes)
    best_max = current_max
    report_best_vrp(best_routes)
    
    # SA parameters
    max_iter = 500 * m
    T0 = current_max * 0.1
    if T0 == 0:
        T0 = 1.0
    T = T0
    cooling_rate = 0.995
    
    for it in range(max_iter):
        # choose move type
        move_type = random.randrange(3)
        new_routes = [list(r) for r in routes]
        success = False
        
        if move_type == 0:  # intra-route 2-opt on a random route
            # select a random route with at least 3 nodes (including depot? route length >=4 for interior nodes)
            candidates = [i for i, r in enumerate(new_routes) if len(r) > 3]
            if candidates:
                ri = random.choice(candidates)
                route = new_routes[ri]
                l = len(route)
                i = random.randrange(1, l-2)
                j = random.randrange(i+1, l-1)
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                if new_route[0] == 0 and new_route[-1] == 0:
                    new_routes[ri] = new_route
                    success = True
        elif move_type == 1:  # inter-route relocate
            # pick a random customer from a route with at least 2 customers, move to another route
            src_idx = None
            cust = None
            pos = None
            for _ in range(10):  # attempt up to 10 times
                ri = random.randrange(truck_count)
                if len(new_routes[ri]) <= 2:
                    continue
                pos_i = random.randrange(1, len(new_routes[ri])-1)
                cust = new_routes[ri][pos_i]
                src_idx = ri
                break
            if src_idx is not None:
                dst_idx = random.randrange(truck_count)
                if dst_idx == src_idx:
                    dst_idx = (dst_idx+1) % truck_count
                dst_route = new_routes[dst_idx]
                pos_dst = random.randrange(1, len(dst_route))
                new_dst = dst_route[:pos_dst] + [cust] + dst_route[pos_dst:]
                new_src = new_routes[src_idx][:pos_i] + new_routes[src_idx][pos_i+1:]
                if new_dst[0]==0 and new_dst[-1]==0 and new_src[0]==0 and new_src[-1]==0:
                    new_routes[src_idx] = new_src
                    new_routes[dst_idx] = new_dst
                    success = True
        else:  # inter-route swap
            # pick two customers from different routes and swap them
            cust1 = None
            cust2 = None
            ri1 = None
            ri2 = None
            pos1 = None
            pos2 = None
            for _ in range(10):
                ri1 = random.randrange(truck_count)
                if len(new_routes[ri1]) <= 2:
                    continue
                ri2 = random.randrange(truck_count)
                if ri2 == ri1 or len(new_routes[ri2]) <= 2:
                    continue
                pos1 = random.randrange(1, len(new_routes[ri1])-1)
                pos2 = random.randrange(1, len(new_routes[ri2])-1)
                cust1 = new_routes[ri1][pos1]
                cust2 = new_routes[ri2][pos2]
                break
            if cust1 is not None:
                new_route1 = new_routes[ri1][:]
                new_route2 = new_routes[ri2][:]
                new_route1[pos1] = cust2
                new_route2[pos2] = cust1
                if new_route1[0]==0 and new_route1[-1]==0 and new_route2[0]==0 and new_route2[-1]==0:
                    new_routes[ri1] = new_route1
                    new_routes[ri2] = new_route2
                    success = True
        
        if not success:
            T *= cooling_rate
            continue
        
        new_max = compute_max(new_routes)
        if new_max < best_max - 1e-9:
            best_max = new_max
            best_routes = [list(r) for r in new_routes]
            report_best_vrp(best_routes)
        
        delta = new_max - current_max
        if delta <= 0 or random.random() < math.exp(-delta/T):
            routes = new_routes
            current_max = new_max
        T *= cooling_rate
    
    return best_routes