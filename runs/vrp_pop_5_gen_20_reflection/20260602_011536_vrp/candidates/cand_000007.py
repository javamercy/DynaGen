import numpy as np
from itertools import permutations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix.tolist()
    # Construction: greedy insertion minimizing max route distance, customers sorted by distance from depot descending
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    customers = list(range(1, n))
    customers.sort(key=lambda c: dist[0][c], reverse=True)
    for c in customers:
        best_max = float('inf')
        best_inc = float('inf')
        best_r = None
        best_pos = None
        for r in range(truck_count):
            route = routes[r]
            cur_dist = route_dists[r]
            for i in range(len(route)-1):
                a = route[i]
                b = route[i+1]
                inc = dist[a][c] + dist[c][b] - dist[a][b]
                new_dist = cur_dist + inc
                other_max = max(route_dists[:r] + route_dists[r+1:] + [0.0])
                new_max = max(other_max, new_dist)
                if (new_max < best_max) or (new_max == best_max and inc < best_inc):
                    best_max = new_max
                    best_inc = inc
                    best_r = r
                    best_pos = i
        routes[best_r].insert(best_pos+1, c)
        route_dists[best_r] += best_inc
    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)

    # Helper functions
    def route_distance(route):
        return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))

    def objective(routes):
        return max(route_distance(r) for r in routes)

    # Local search: focused on the longest route
    max_passes = 10
    for _ in range(max_passes):
        improved = False
        # Identify route with maximum distance
        max_dist = 0
        max_idx = 0
        for r, rd in enumerate(route_dists):
            if rd > max_dist:
                max_dist = rd
                max_idx = r
        # Focused improvement on max_idx route
        route = routes[max_idx]
        # 2-opt on that route
        if len(route) > 3:
            for i in range(1, len(route)-2):
                for k in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < route_dists[max_idx]:
                        routes[max_idx] = new_route
                        route_dists[max_idx] = new_dist
                        current_max = max(route_dists)
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [list(r2) for r2 in routes]
                            report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
        if improved:
            continue
        # Relocate from max_idx route to others
        # Iterate over customers in the longest route (excluding depots)
        for pos in range(1, len(route)-1):
            node = route[pos]
            # Try moving to other routes
            best_new_max = objective(routes)  # current objective
            best_r = None
            best_pos = None
            for r2 in range(truck_count):
                if r2 == max_idx:
                    continue
                route2 = routes[r2]
                for p2 in range(1, len(route2)):
                    new_route_max_idx = route[:pos] + route[pos+1:]
                    if len(new_route_max_idx) < 2:
                        new_route_max_idx = [0, 0]
                    new_route_r2 = route2[:p2] + [node] + route2[p2:]
                    new_dist_max = route_distance(new_route_max_idx)
                    new_dist_r2 = route_distance(new_route_r2)
                    other_dists = [route_dists[i] for i in range(truck_count) if i != max_idx and i != r2]
                    new_max = max([new_dist_max, new_dist_r2] + other_dists)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_r = r2
                        best_pos = p2
            if best_r is not None:
                # Apply move
                del routes[max_idx][pos]
                if len(routes[max_idx]) < 2:
                    routes[max_idx] = [0, 0]
                routes[best_r].insert(best_pos, node)
                route_dists[max_idx] = route_distance(routes[max_idx])
                route_dists[best_r] = route_distance(routes[best_r])
                current_max = max(route_dists)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [list(r2) for r2 in routes]
                    report_best_vrp(best_routes)
                improved = True
                break
        if improved:
            continue
        # Global relocate (first improvement, iterate nodes in order)
        for node in range(1, n):
            # find current route and position
            curr_r = None
            curr_pos = None
            for r, route in enumerate(routes):
                for p, cust in enumerate(route):
                    if cust == node:
                        curr_r = r
                        curr_pos = p
                        break
                if curr_r is not None:
                    break
            if curr_r is None:
                continue
            # try other routes and positions
            best_new_obj = objective(routes)
            best_r = None
            best_pos = None
            for r2 in range(truck_count):
                if r2 == curr_r:
                    continue
                route2 = routes[r2]
                for p2 in range(1, len(route2)):
                    new_route_curr = routes[curr_r][:curr_pos] + routes[curr_r][curr_pos+1:]
                    if len(new_route_curr) < 2:
                        new_route_curr = [0, 0]
                    new_route_r2 = route2[:p2] + [node] + route2[p2:]
                    new_dist_curr = route_distance(new_route_curr)
                    new_dist_r2 = route_distance(new_route_r2)
                    other_dists = [route_dists[i] for i in range(truck_count) if i != curr_r and i != r2]
                    new_max = max([new_dist_curr, new_dist_r2] + other_dists)
                    if new_max < best_new_obj:
                        best_new_obj = new_max
                        best_r = r2
                        best_pos = p2
                        # We break inner loops in first improvement, but keep scanning to find first?
                        # Actually we want first improvement, so we break after finding first?
                        # But we are inside loops; we'll break out after applying.
            if best_r is not None:
                # Apply move
                del routes[curr_r][curr_pos]
                if len(routes[curr_r]) < 2:
                    routes[curr_r] = [0, 0]
                routes[best_r].insert(best_pos, node)
                route_dists[curr_r] = route_distance(routes[curr_r])
                route_dists[best_r] = route_distance(routes[best_r])
                current_max = max(route_dists)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [list(r2) for r2 in routes]
                    report_best_vrp(best_routes)
                improved = True
                break  # break node loop after first improvement
        if improved:
            continue
        # Global swap (pairs of nodes from different routes)
        # iterate nodes in order
        nodes = list(range(1, n))
        for i in range(len(nodes)):
            node_i = nodes[i]
            # find route and position
            ri = None
            pi = None
            for r, route in enumerate(routes):
                for p, cust in enumerate(route):
                    if cust == node_i:
                        ri = r
                        pi = p
                        break
                if ri is not None:
                    break
            if ri is None:
                continue
            for j in range(i+1, len(nodes)):
                node_j = nodes[j]
                rj = None
                pj = None
                for r, route in enumerate(routes):
                    for p, cust in enumerate(route):
                        if cust == node_j:
                            rj = r
                            pj = p
                            break
                    if rj is not None:
                        break
                if rj is None or rj == ri:
                    continue
                # Remove both
                route_i_without = routes[ri][:pi] + routes[ri][pi+1:]
                if len(route_i_without) < 2:
                    route_i_without = [0, 0]
                route_j_without = routes[rj][:pj] + routes[rj][pj+1:]
                if len(route_j_without) < 2:
                    route_j_without = [0, 0]
                # Try insertions
                best_new_obj_swap = objective(routes)
                best_pi = None
                best_pj = None
                for p_i in range(1, len(route_i_without)):
                    for p_j in range(1, len(route_j_without)):
                        new_route_i = route_i_without[:p_i] + [node_j] + route_i_without[p_i:]
                        new_route_j = route_j_without[:p_j] + [node_i] + route_j_without[p_j:]
                        new_dist_i = route_distance(new_route_i)
                        new_dist_j = route_distance(new_route_j)
                        other_dists = [route_dists[k] for k in range(truck_count) if k != ri and k != rj]
                        new_max = max([new_dist_i, new_dist_j] + other_dists)
                        if new_max < best_new_obj_swap:
                            best_new_obj_swap = new_max
                            best_pi = p_i
                            best_pj = p_j
                if best_pi is not None:
                    # Apply swap
                    routes[ri] = route_i_without[:best_pi] + [node_j] + route_i_without[best_pi:]
                    routes[rj] = route_j_without[:best_pj] + [node_i] + route_j_without[best_pj:]
                    route_dists[ri] = route_distance(routes[ri])
                    route_dists[rj] = route_distance(routes[rj])
                    current_max = max(route_dists)
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [list(r2) for r2 in routes]
                        report_best_vrp(best_routes)
                    improved = True
                    break  # break j loop
            if improved:
                break
        if not improved:
            break
    if max(route_dists) < best_max:
        best_routes = [list(r) for r in routes]
    return best_routes