def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    routes = [[0,0] for _ in range(truck_count)]
    route_dists = [0] * truck_count
    customers = list(range(1, n))
    customers.sort(key=lambda c: distance_matrix[0][c], reverse=True)
    for c in customers:
        best_new_max = float('inf')
        best_increase = float('inf')
        best_r = -1
        best_pos = -1
        for r in range(truck_count):
            route = routes[r]
            current_dist = route_dists[r]
            for i in range(len(route)-1):
                a = route[i]
                b = route[i+1]
                increase = distance_matrix[a][c] + distance_matrix[c][b] - distance_matrix[a][b]
                new_dist = current_dist + increase
                other_max = max(route_dists[:r] + route_dists[r+1:])
                new_max = max(other_max, new_dist)
                if (new_max < best_new_max) or \
                   (new_max == best_new_max and increase < best_increase) or \
                   (new_max == best_new_max and increase == best_increase and r < best_r) or \
                   (new_max == best_new_max and increase == best_increase and r == best_r and i < best_pos):
                    best_new_max = new_max
                    best_increase = increase
                    best_r = r
                    best_pos = i
        routes[best_r].insert(best_pos+1, c)
        route_dists[best_r] += best_increase
    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)
    # Intra-route Or-opt improvement
    for r in range(truck_count):
        route = routes[r]
        improved = True
        iter_count = 0
        max_iter = 100
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            found = False
            for seg_len in range(1, min(4, len(route)-2)):
                if found: break
                for i in range(1, len(route)-seg_len):
                    if found: break
                    segment = route[i:i+seg_len]
                    new_route_no_seg = route[:i] + route[i+seg_len:]
                    for j in range(1, len(new_route_no_seg)):
                        candidate = new_route_no_seg[:j] + segment + new_route_no_seg[j:]
                        new_dist = 0
                        for k in range(len(candidate)-1):
                            new_dist += distance_matrix[candidate[k]][candidate[k+1]]
                        if new_dist < route_dists[r]:
                            routes[r] = candidate
                            route_dists[r] = new_dist
                            improved = True
                            found = True
                            current_max = max(route_dists)
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [list(r2) for r2 in routes]
                                report_best_vrp(best_routes)
                            break
                    if found: break
                if found: break
            route = routes[r]
    final_max = max(route_dists)
    if final_max < best_max:
        best_routes = [list(r) for r in routes]
    return best_routes