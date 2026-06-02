import numpy as np

def solve_vrp(distance_matrix, truck_count):
    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    # Regret-based greedy insertion
    def construct():
        routes = [[0, 0] for _ in range(truck_count)]
        dists = [0.0] * truck_count
        unassigned = set(customers)
        
        while unassigned:
            best_cust = None
            best_regret = -1.0
            best_truck = None
            best_pos = None
            best_increase = None
            for cust in unassigned:
                increases = []
                for t in range(truck_count):
                    route = routes[t]
                    best_inc = float('inf')
                    best_pos_t = None
                    for pos in range(1, len(route)):
                        new_dist = dists[t] \
                            + distance_matrix[route[pos-1], cust] \
                            + distance_matrix[cust, route[pos]] \
                            - distance_matrix[route[pos-1], route[pos]]
                        other_dists = [dists[i] for i in range(truck_count) if i != t]
                        new_max = max(new_dist, max(other_dists) if other_dists else 0.0)
                        increase = new_max - max(dists)
                        if increase < best_inc or (increase == best_inc and cust < best_cust_global):
                            best_inc = increase
                            best_pos_t = pos
                    if best_inc < float('inf'):
                        increases.append((t, best_pos_t, best_inc))
                if len(increases) < 2:
                    regret = 0.0
                else:
                    increases_sorted = sorted(increases, key=lambda x: x[2])
                    regret = increases_sorted[1][2] - increases_sorted[0][2]
                if regret > best_regret or (regret == best_regret and cust < best_cust):
                    best_regret = regret
                    best_cust = cust
                    best_truck = increases_sorted[0][0]
                    best_pos = increases_sorted[0][1]
                    best_increase = increases_sorted[0][2]
            # Insert
            route = routes[best_truck]
            route.insert(best_pos, best_cust)
            dists[best_truck] = route_distance(route)
            unassigned.remove(best_cust)
        return routes, dists

    # Initialize
    trucks, dists = construct()
    best_routes = [list(r) for r in trucks]
    best_max = max(dists)
    report_best_vrp(best_routes)

    def eval_max(t_idx, new_route):
        new_dist = route_distance(new_route)
        other_dists = [dists[i] for i in range(truck_count) if i != t_idx]
        return max(new_dist, max(other_dists) if other_dists else 0.0)

    # 2-opt on each route
    for t in range(truck_count):
        route = trucks[t]
        improved = True
        iteration = 0
        max_iter = len(route) * len(route)
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if eval_max(t, new_route) < max(dists):
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
    iteration = 0
    max_iter = n * truck_count
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

    # Swap customers between routes
    iteration = 0
    max_iter = n * truck_count
    improved = True
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for src in range(truck_count):
            route_src = trucks[src]
            for pos_src in range(1, len(route_src)-1):
                cust_src = route_src[pos_src]
                for dst in range(truck_count):
                    if dst <= src:
                        continue
                    route_dst = trucks[dst]
                    for pos_dst in range(1, len(route_dst)-1):
                        cust_dst = route_dst[pos_dst]
                        # Swap cust_src and cust_dst
                        new_route_src = route_src[:pos_src] + [cust_dst] + route_src[pos_src+1:]
                        new_route_dst = route_dst[:pos_dst] + [cust_src] + route_dst[pos_dst+1:]
                        new_dist_src = route_distance(new_route_src)
                        new_dist_dst = route_distance(new_route_dst)
                        other_dists = [dists[i] for i in range(truck_count) if i != src and i != dst]
                        new_max = max(new_dist_src, new_dist_dst, max(other_dists) if other_dists else 0.0)
                        if new_max < max(dists):
                            trucks[src] = new_route_src
                            dists[src] = new_dist_src
                            trucks[dst] = new_route_dst
                            dists[dst] = new_dist_dst
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