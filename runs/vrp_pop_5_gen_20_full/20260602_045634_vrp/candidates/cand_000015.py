import numpy as np

def solve_vrp(distance_matrix, truck_count):
    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    n = distance_matrix.shape[0]
    trucks = [[0, 0] for _ in range(truck_count)]
    dists = [0.0] * truck_count
    customers = list(range(1, n))

    # Greedy insertion
    for cust in customers:
        best_increase = float('inf')
        best_truck = -1
        best_pos = -1
        for t in range(truck_count):
            route = trucks[t]
            for pos in range(1, len(route)):
                new_dist = dists[t] \
                    + distance_matrix[route[pos-1], cust] \
                    + distance_matrix[cust, route[pos]] \
                    - distance_matrix[route[pos-1], route[pos]]
                new_max = max(new_dist, max(dists[:t] + dists[t+1:]))
                increase = new_max - max(dists)
                # Deterministic tie-breaking: smaller increase, then smaller customer index, then smaller truck index
                if (increase < best_increase) or (increase == best_increase and cust < best_cust) or (increase == best_increase and cust == best_cust and t < best_truck):
                    best_increase = increase
                    best_truck = t
                    best_pos = pos
                    best_cust = cust
        # Insert
        route = trucks[best_truck]
        route.insert(best_pos, cust)
        dists[best_truck] = route_distance(route)

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
                    if eval_max_mod(t, new_route) < max(dists):
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