import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix.tolist()

    def route_distance(route):
        return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))

    def objective(routes):
        return max(route_distance(r) for r in routes)

    def cheapest_insertion_maxmin(routes, unassigned):
        """Insert all unassigned customers using cheapest insertion minimizing max route distance."""
        while unassigned:
            best_max = float('inf')
            best_total = float('inf')
            best_node = None
            best_route = None
            best_pos = None
            for node in unassigned:
                for r in range(truck_count):
                    route = routes[r]
                    for pos in range(1, len(route)):
                        # new route distance for route r if insert node at pos
                        new_dist = 0
                        prev = route[0]
                        for k in range(1, len(route)):
                            if k == pos:
                                new_dist += dist[prev][node]
                                prev = node
                            new_dist += dist[prev][route[k]]
                            prev = route[k]
                        new_route_dist = new_dist
                        # compute new max across all routes
                        current_max = 0
                        for rr in range(truck_count):
                            if rr == r:
                                route_dist = new_route_dist
                            else:
                                route_dist = route_distance(routes[rr])
                            if route_dist > current_max:
                                current_max = route_dist
                        if current_max < best_max or (current_max == best_max and new_route_dist < best_total):
                            best_max = current_max
                            best_total = new_route_dist
                            best_node = node
                            best_route = r
                            best_pos = pos
            routes[best_route].insert(best_pos, best_node)
            unassigned.remove(best_node)

    # Construction
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    cheapest_insertion_maxmin(routes, unassigned)

    best_obj = objective(routes)
    best_routes = [list(r) for r in routes]
    report_best_vrp([list(r) for r in routes])

    def local_search(routes):
        """First-improvement local search until no improvement or max passes."""
        max_passes = 20
        for _ in range(max_passes):
            improved = False
            # Relocate
            for node in range(1, n):
                curr_route = None
                curr_pos = None
                for r, route in enumerate(routes):
                    for pos, cust in enumerate(route):
                        if cust == node:
                            curr_route = r
                            curr_pos = pos
                            break
                    if curr_route is not None:
                        break
                if curr_route is None:
                    continue
                best_new_obj = float('inf')
                best_r = None
                best_pos = None
                for r in range(truck_count):
                    if r == curr_route:
                        continue
                    route = routes[r]
                    for pos in range(1, len(route)):
                        new_route_r = route[:pos] + [node] + route[pos:]
                        new_route_curr = routes[curr_route][:curr_pos] + routes[curr_route][curr_pos+1:]
                        if len(new_route_curr) < 2:
                            new_route_curr = [0, 0]
                        new_routes = [list(routes[i]) for i in range(truck_count)]
                        new_routes[r] = new_route_r
                        new_routes[curr_route] = new_route_curr
                        obj = max(route_distance(rr) for rr in new_routes)
                        if obj < best_new_obj:
                            best_new_obj = obj
                            best_r = r
                            best_pos = pos
                if best_new_obj < objective(routes):
                    routes[curr_route].pop(curr_pos)
                    if len(routes[curr_route]) < 2:
                        routes[curr_route] = [0, 0]
                    routes[best_r].insert(best_pos, node)
                    improved = True
                    break
            if improved:
                continue
            # Swap
            for i in range(1, n):
                ri = None
                pi = None
                for r, route in enumerate(routes):
                    for p, cust in enumerate(route):
                        if cust == i:
                            ri = r
                            pi = p
                            break
                    if ri is not None:
                        break
                if ri is None:
                    continue
                for j in range(i+1, n):
                    rj = None
                    pj = None
                    for r, route in enumerate(routes):
                        for p, cust in enumerate(route):
                            if cust == j:
                                rj = r
                                pj = p
                                break
                        if rj is not None:
                            break
                    if rj is None or rj == ri:
                        continue
                    route_i_without = routes[ri][:pi] + routes[ri][pi+1:]
                    if len(route_i_without) < 2:
                        route_i_without = [0, 0]
                    route_j_without = routes[rj][:pj] + routes[rj][pj+1:]
                    if len(route_j_without) < 2:
                        route_j_without = [0, 0]
                    best_obj_swap = float('inf')
                    best_pos_i = None
                    best_pos_j = None
                    for pos_i in range(1, len(route_i_without)):
                        for pos_j in range(1, len(route_j_without)):
                            new_route_i = route_i_without[:pos_i] + [j] + route_i_without[pos_i:]
                            new_route_j = route_j_without[:pos_j] + [i] + route_j_without[pos_j:]
                            new_routes = [list(routes[k]) for k in range(truck_count)]
                            new_routes[ri] = new_route_i
                            new_routes[rj] = new_route_j
                            obj = max(route_distance(rr) for rr in new_routes)
                            if obj < best_obj_swap:
                                best_obj_swap = obj
                                best_pos_i = pos_i
                                best_pos_j = pos_j
                    if best_obj_swap < objective(routes):
                        routes[ri] = route_i_without[:best_pos_i] + [j] + route_i_without[best_pos_i:]
                        routes[rj] = route_j_without[:best_pos_j] + [i] + route_j_without[best_pos_j:]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # 2-opt within each route
            for r in range(truck_count):
                route = routes[r]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        old_dist = route_distance(route)
                        other_max = 0
                        for rr in range(truck_count):
                            if rr != r:
                                d = route_distance(routes[rr])
                                if d > other_max:
                                    other_max = d
                        new_max = max(new_dist, other_max)
                        if new_max < objective(routes):
                            routes[r] = new_route
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            if not improved:
                break

    def perturb_and_restart(routes):
        """Perturb by removing customers at odd positions (excluding depots) and reinserting."""
        removed = []
        for r in range(truck_count):
            route = routes[r]
            new_route = [0]
            for idx in range(1, len(route)-1):
                cust = route[idx]
                if idx % 2 == 1:  # odd positions (1-indexed)
                    removed.append(cust)
                else:
                    new_route.append(cust)
            new_route.append(0)
            routes[r] = new_route
        # Reinsert all removed customers using cheapest insertion
        cheapest_insertion_maxmin(routes, removed)

    # Local search on initial solution
    local_search(routes)
    if objective(routes) < best_obj:
        best_obj = objective(routes)
        best_routes = [list(r) for r in routes]
        report_best_vrp([list(r) for r in routes])

    # Restart rounds
    for restart in range(3):
        # Create a copy to perturb
        new_routes = [list(r) for r in best_routes]
        perturb_and_restart(new_routes)
        local_search(new_routes)
        if objective(new_routes) < best_obj:
            best_obj = objective(new_routes)
            best_routes = [list(r) for r in new_routes]
            report_best_vrp([list(r) for r in new_routes])
        else:
            # optional: break if no improvement after few restarts
            pass

    return [list(r) for r in best_routes]