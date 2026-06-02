import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix
    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0 for _ in range(truck_count)]
    unassigned = set(range(1, n))
    
    def compute_route_length(route):
        if len(route) <= 2:
            return 2 * dist[0, route[1]] if len(route) == 2 else 0.0
        length = 0.0
        for i in range(len(route) - 1):
            length += dist[route[i], route[i+1]]
        return length
    
    while unassigned:
        best_regret = -1.0
        best_customer = None
        best_route = -1
        best_pos = -1
        for c in unassigned:
            insertions = []
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_len = route_lengths[r_idx] - dist[route[pos-1], route[pos]] + dist[route[pos-1], c] + dist[c, route[pos]]
                    other_max = max(route_lengths[:r_idx] + route_lengths[r_idx+1:], default=0.0)
                    new_max = max(new_len, other_max)
                    insertions.append((new_max, r_idx, pos))
            insertions.sort(key=lambda x: x[0])
            if len(insertions) >= 3:
                regret = (insertions[1][0] - insertions[0][0]) + (insertions[2][0] - insertions[0][0])
            elif len(insertions) == 2:
                regret = insertions[1][0] - insertions[0][0]
            else:
                regret = 0.0
            if regret > best_regret or (regret == best_regret and best_customer is not None and c < best_customer):
                best_regret = regret
                best_customer = c
                best_route, best_pos = insertions[0][1], insertions[0][2]
        if best_customer is None:
            break
        routes[best_route].insert(best_pos, best_customer)
        route_lengths[best_route] = compute_route_length(routes[best_route])
        unassigned.remove(best_customer)
    
    current_routes = [list(r) for r in routes]
    current_max = max(route_lengths)
    report_best_vrp(current_routes)
    
    # intra-route 2-opt initial
    for r_idx in range(truck_count):
        route = current_routes[r_idx]
        improved = True
        it = 0
        while improved and it < 10 * n:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_len = compute_route_length(new_route)
                    if new_len < route_lengths[r_idx]:
                        route_lengths[r_idx] = new_len
                        current_routes[r_idx] = new_route
                        improved = True
                        if max(route_lengths) < current_max:
                            current_max = max(route_lengths)
                            report_best_vrp(current_routes)
                        break
                if improved:
                    break
            it += 1
    
    # inter-route improvement (swap and relocate) plus intra-2opt
    max_global_iter = n * truck_count
    global_iter = 0
    improved = True
    while improved and global_iter < max_global_iter:
        improved = False
        global_iter += 1
        # evaluate all swap moves
        best_new_max = current_max
        best_move = None
        for r1 in range(truck_count):
            for r2 in range(r1+1, truck_count):
                route1 = current_routes[r1]
                route2 = current_routes[r2]
                for i in range(1, len(route1)-1):
                    for j in range(1, len(route2)-1):
                        new1 = route1[:i] + [route2[j]] + route1[i+1:]
                        new2 = route2[:j] + [route1[i]] + route2[j+1:]
                        new_len1 = compute_route_length(new1)
                        new_len2 = compute_route_length(new2)
                        other = [route_lengths[t] for t in range(truck_count) if t != r1 and t != r2]
                        other_max = max(other) if other else 0.0
                        new_max = max(new_len1, new_len2, other_max)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('swap', r1, i, r2, j)
        # evaluate all relocate moves
        for r1 in range(truck_count):
            for i in range(1, len(current_routes[r1])-1):
                c = current_routes[r1][i]
                for r2 in range(truck_count):
                    if r2 == r1:
                        continue
                    route2 = current_routes[r2]
                    for j in range(1, len(route2)):
                        new1 = current_routes[r1][:i] + current_routes[r1][i+1:]
                        new2 = route2[:j] + [c] + route2[j:]
                        new_len1 = compute_route_length(new1)
                        new_len2 = compute_route_length(new2)
                        other = [route_lengths[t] for t in range(truck_count) if t != r1 and t != r2]
                        other_max = max(other) if other else 0.0
                        new_max = max(new_len1, new_len2, other_max)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('relocate', r1, i, r2, j)
        if best_move is not None:
            improved = True
            if best_move[0] == 'swap':
                _, r1, i, r2, j = best_move
                # swap nodes
                current_routes[r1][i], current_routes[r2][j] = current_routes[r2][j], current_routes[r1][i]
                route_lengths[r1] = compute_route_length(current_routes[r1])
                route_lengths[r2] = compute_route_length(current_routes[r2])
            else:
                _, r1, i, r2, j = best_move
                c = current_routes[r1].pop(i)
                current_routes[r2].insert(j, c)
                route_lengths[r1] = compute_route_length(current_routes[r1])
                route_lengths[r2] = compute_route_length(current_routes[r2])
            current_max = best_new_max
            report_best_vrp(current_routes)
            # optional intra 2-opt on affected routes
            for r_idx in [r1, r2]:
                route = current_routes[r_idx]
                improved2 = True
                it2 = 0
                while improved2 and it2 < 10 * n:
                    improved2 = False
                    for ii in range(1, len(route)-2):
                        for jj in range(ii+1, len(route)-1):
                            new_route = route[:ii] + route[ii:jj+1][::-1] + route[jj+1:]
                            new_len = compute_route_length(new_route)
                            if new_len < route_lengths[r_idx]:
                                route_lengths[r_idx] = new_len
                                current_routes[r_idx] = new_route
                                improved2 = True
                                if max(route_lengths) < current_max:
                                    current_max = max(route_lengths)
                                    report_best_vrp(current_routes)
                                break
                        if improved2:
                            break
                    it2 += 1
    return current_routes