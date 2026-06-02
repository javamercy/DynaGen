import numpy as np

def solve_vrp(distance_matrix, truck_count):
    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    n = distance_matrix.shape[0]
    trucks = [[0,0] for _ in range(truck_count)]
    dists = [0.0] * truck_count
    customers = list(range(1, n))
    assigned = [False] * (n-1)
    unassigned = list(range(1,n))

    # Insert first customer to each truck to avoid all empty? Actually we can just start with empty and insert one by one.
    # To handle regret correctly, we need at least one customer per route? No, but regret calculation works with empty routes too.
    # We'll insert customers one by one using regret.
    
    while unassigned:
        best_cust = None
        best_regret = -float('inf')
        best_truck = -1
        best_pos = -1
        best_max_after = float('inf')
        for cust in unassigned:
            # Compute insertions for each truck and position
            increases = []  # for each possible insertion: (increase, truck, pos)
            for t in range(truck_count):
                route = trucks[t]
                for pos in range(1, len(route)):
                    new_dist = dists[t] \
                        + distance_matrix[route[pos-1], cust] \
                        + distance_matrix[cust, route[pos]] \
                        - distance_matrix[route[pos-1], route[pos]]
                    new_max = max(new_dist, max([dists[i] for i in range(truck_count) if i!=t] or [0.0]))
                    increase = new_max - max(dists)
                    increases.append((increase, t, pos))
                # Also consider inserting after last? Already covered because pos goes up to len(route)-1 which is last 0? Actually route ends with 0, so pos from 1 to len(route)-1 includes before each stop and before returning depot? Typical: positions where you can insert between consecutive nodes. Since route ends with 0, len(route)-1 is the index of last 0? Let's adjust: route = [0, ..., 0], indices 0 and len(route)-1 are depots. Insert at positions 1..len(route)-2? Actually we want to insert after depot and before the end. The loop for pos in range(1, len(route)): will go from index 1 to last index (which is the last 0). That would allow insertion between last customer and depot. That's valid. But we need to ensure we don't insert after the end? It's fine. However empty route [0,0] has len=2, so pos=1 only, inserting between the two zeros, which is correct.
            if not increases:
                # Should not happen because at least one truck has at least one insertion position
                continue
            # Sort by increase, then by truck, then by position
            increases.sort(key=lambda x: (x[0], cust, x[1], x[2]))
            best_inc = increases[0][0]
            second_inc = increases[1][0] if len(increases)>1 else best_inc
            regret = second_inc - best_inc
            # Choose customer with highest regret, tie-break by smaller cust then smaller best_inc
            if (regret > best_regret) or (regret == best_regret and cust < best_cust) or (regret == best_regret and cust == best_cust and best_inc < best_max_after):
                best_regret = regret
                best_cust = cust
                best_truck = increases[0][1]
                best_pos = increases[0][2]
                best_max_after = max(dists) + best_inc  # not needed exactly
        # Insert best_cust
        route = trucks[best_truck]
        route.insert(best_pos, best_cust)
        dists[best_truck] = route_distance(route)
        unassigned.remove(best_cust)

    best_routes = [list(r) for r in trucks]
    best_max = max(dists)
    report_best_vrp(best_routes)

    def eval_max_mod(t_idx, new_route):
        new_dist = route_distance(new_route)
        other_dists = [dists[i] for i in range(truck_count) if i != t_idx]
        return max(new_dist, max(other_dists) if other_dists else 0.0)

    # 2-opt for each route
    for t in range(truck_count):
        route = trucks[t]
        improved = True
        iteration = 0
        max_iter = len(route) * 2
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_max = eval_max_mod(t, new_route)
                    if new_max < max(dists):
                        trucks[t] = new_route
                        dists[t] = route_distance(new_route)
                        improved = True
                        if max(dists) < best_max:
                            best_routes = [list(r) for r in trucks]
                            best_max = max(dists)
                            report_best_vrp(best_routes)
                        break
                if improved:
                    break

    # Relocate
    max_iter = n * truck_count
    iteration = 0
    improved = True
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for src in range(truck_count):
            route_src = trucks[src]
            for pos_src in range(1, len(route_src)-1):
                cust = route_src[pos_src]
                temp_src = route_src[:pos_src] + route_src[pos_src+1:]
                new_dist_src = route_distance(temp_src)
                for dst in range(truck_count):
                    if dst == src:
                        continue
                    route_dst = trucks[dst]
                    for pos_dst in range(1, len(route_dst)):
                        new_route_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                        other_dists = [dists[i] for i in range(truck_count) if i != src and i != dst]
                        new_max = max(new_dist_src, route_distance(new_route_dst), max(other_dists) if other_dists else 0.0)
                        if new_max < max(dists):
                            trucks[src] = temp_src
                            dists[src] = new_dist_src
                            trucks[dst] = new_route_dst
                            dists[dst] = route_distance(new_route_dst)
                            improved = True
                            if max(dists) < best_max:
                                best_routes = [list(r) for r in trucks]
                                best_max = max(dists)
                                report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return best_routes