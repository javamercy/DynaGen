import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    # initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    route_distances = [0.0 for _ in range(truck_count)]
    
    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    for r in range(truck_count):
        route_distances[r] = compute_route_distance(routes[r])
    
    unassigned = set(range(1, n))
    
    def best_max(customer):
        best_val = float('inf')
        best_pos = None
        second_val = float('inf')
        for r_idx, route in enumerate(routes):
            curr_dist = route_distances[r_idx]
            for i in range(1, len(route)):
                new_dist = curr_dist - distance_matrix[route[i-1], route[i]] \
                           + distance_matrix[route[i-1], customer] \
                           + distance_matrix[customer, route[i]]
                other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0.0)
                cand_max = max(new_dist, other_max)
                if cand_max < best_val - 1e-12:
                    second_val = best_val
                    best_val = cand_max
                    best_pos = (r_idx, i)
                elif cand_max < second_val - 1e-12 and abs(cand_max - best_val) > 1e-12:
                    second_val = cand_max
        return best_val, second_val, best_pos
    
    # Construction: regret-2
    while unassigned:
        regrets = []
        for c in unassigned:
            best_val, second_val, _ = best_max(c)
            regret = second_val - best_val if second_val != float('inf') else 0.0
            regrets.append((regret, best_val, c))
        regrets.sort(key=lambda x: (-x[0], -x[1], x[2]))
        selected = regrets[0][2]
        _, _, best_pos = best_max(selected)
        r_idx, i = best_pos
        route = routes[r_idx]
        route.insert(i, selected)
        route_distances[r_idx] = compute_route_distance(route)
        unassigned.remove(selected)
    
    current_routes = [list(r) for r in routes]
    current_max = max(route_distances)
    best_routes = [list(r) for r in current_routes]
    best_max = current_max
    
    def report_best_vrp(routes):
        pass
    
    def improve(routes, route_dists):
        # local search: 2-opt, swap, relocate
        # returns (routes, route_dists, improved_flag)
        improved = True
        max_iters = n * n
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            # intra-route 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                best_improve = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_route_distance(new_route)
                        if new_dist < route_dists[r_idx] - 1e-12:
                            route_dists[r_idx] = new_dist
                            routes[r_idx] = new_route
                            improved = True
                            best_improve = True
                            break
                    if best_improve:
                        break
                if best_improve:
                    continue
            # inter-route swap
            for r1 in range(truck_count):
                for r2 in range(r1+1, truck_count):
                    route1 = routes[r1]
                    route2 = routes[r2]
                    best_improve = False
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i] + [route2[j]] + route1[i+1:]
                            new2 = route2[:j] + [route1[i]] + route2[j+1:]
                            new_dist1 = compute_route_distance(new1)
                            new_dist2 = compute_route_distance(new2)
                            other_max = max([route_dists[k] for k in range(truck_count) if k != r1 and k != r2] + [new_dist1, new_dist2])
                            if other_max < current_max - 1e-12:
                                routes[r1] = new1
                                routes[r2] = new2
                                route_dists[r1] = new_dist1
                                route_dists[r2] = new_dist2
                                current_max = other_max
                                improved = True
                                best_improve = True
                                break
                        if best_improve:
                            break
                    if best_improve:
                        break
                if best_improve:
                    break
            if improved:
                continue
            # inter-route relocate
            for r1 in range(truck_count):
                for r2 in range(truck_count):
                    if r1 == r2:
                        continue
                    route1 = routes[r1]
                    route2 = routes[r2]
                    best_improve = False
                    for i in range(1, len(route1)-1):
                        c = route1[i]
                        new1 = route1[:i] + route1[i+1:]
                        new_dist1 = compute_route_distance(new1)
                        # find best insertion in route2
                        best_new2 = None
                        best_new_dist2 = float('inf')
                        for j in range(1, len(route2)):
                            new2 = route2[:j] + [c] + route2[j:]
                            d2 = compute_route_distance(new2)
                            if d2 < best_new_dist2:
                                best_new_dist2 = d2
                                best_new2 = new2
                        other_max = max(route_dists[:r1] + route_dists[r1+1:r2] + route_dists[r2+1:], default=0.0)
                        cand_max = max(new_dist1, best_new_dist2, other_max)
                        if cand_max < current_max - 1e-12:
                            routes[r1] = new1
                            routes[r2] = best_new2
                            route_dists[r1] = new_dist1
                            route_dists[r2] = best_new_dist2
                            current_max = cand_max
                            improved = True
                            best_improve = True
                            break
                    if best_improve:
                        break
                if best_improve:
                    break
        return routes, route_dists, improved

    # initial improvement
    current_routes, route_distances, _ = improve(current_routes, route_distances)
    current_max = max(route_distances)
    if current_max < best_max:
        best_max = current_max
        best_routes = [list(r) for r in current_routes]
        report_best_vrp(best_routes)

    # deterministic restart
    max_restarts = 5
    for restart in range(max_restarts):
        # perturb: remove few customers from the route with maximum distance
        max_route_idx = np.argmax(route_distances)
        route_to_perturb = current_routes[max_route_idx]
        if len(route_to_perturb) <= 3:
            continue  # cannot remove non-depot customers
        # remove 2 customers (or less if route is short)
        num_remove = min(2, len(route_to_perturb) - 2)
        # remove the first num_remove customers (excluding depot) to be deterministic
        remove_indices = sorted([i for i in range(1, len(route_to_perturb)-1)])[:num_remove]
        removed_customers = [route_to_perturb[i] for i in remove_indices]
        # rebuild route without those customers
        new_route = [0] + [route_to_perturb[i] for i in range(1, len(route_to_perturb)-1) if i not in remove_indices] + [0]
        routes = [list(r) for r in current_routes]
        routes[max_route_idx] = new_route
        route_dists = [compute_route_distance(r) for r in routes]
        # reinsert removed customers using regret-2 (same as construction but only for those)
        unassigned = set(removed_customers)
        # temporarily set global routes to current
        global_routes = routes
        global_route_dists = route_dists
        while unassigned:
            regrets = []
            for c in unassigned:
                best_val, second_val, _ = best_max(c)
                regret = second_val - best_val if second_val != float('inf') else 0.0
                regrets.append((regret, best_val, c))
            regrets.sort(key=lambda x: (-x[0], -x[1], x[2]))
            selected = regrets[0][2]
            _, _, best_pos = best_max(selected)
            r_idx, i = best_pos
            route = routes[r_idx]
            route.insert(i, selected)
            route_dists[r_idx] = compute_route_distance(route)
            unassigned.remove(selected)
        # apply improvement on perturbed solution
        routes, route_dists, _ = improve(routes, route_dists)
        new_max = max(route_dists)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]
            current_routes = best_routes
            route_distances = route_dists
            current_max = best_max
            report_best_vrp(best_routes)
        else:
            # if not improved, revert to best for next restart
            current_routes = [list(r) for r in best_routes]
            route_distances = [compute_route_distance(r) for r in best_routes]
            current_max = best_max

    return best_routes