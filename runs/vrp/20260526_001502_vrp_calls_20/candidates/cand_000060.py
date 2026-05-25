import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]

    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def compute_insertion_cost(route, pos, cust):
        return distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]

    def compute_new_max(routes, route_dists, cust, r, pos):
        inc = compute_insertion_cost(routes[r], pos, cust)
        new_dist = route_dists[r] + inc
        other_max = max(route_dists[i] for i in range(truck_count) if i != r)
        return max(new_dist, other_max)

    def construct_regret_balancing(lambda_weight):
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0 for _ in range(truck_count)]
        unassigned = list(range(1, n))
        while unassigned:
            # For each unassigned customer, compute regret-2 value
            best_regret = -float('inf')
            best_cust = None
            best_r = None
            best_pos = None
            for cust in unassigned:
                costs = []
                for r in range(truck_count):
                    route = routes[r]
                    for pos in range(1, len(route)):
                        inc = compute_insertion_cost(route, pos, cust)
                        new_dist = route_dists[r] + inc
                        # Balancing penalty: delta = max(0, new_dist - avg_dist * lambda_weight)
                        avg_dist = sum(route_dists) / truck_count
                        penalty = max(0, new_dist - avg_dist * lambda_weight)
                        total = inc + penalty
                        costs.append((total, r, pos, new_dist))
                # Sort by total cost, pick best (smallest) and second best
                costs.sort(key=lambda x: x[0])
                best_cost = costs[0][0]
                second_best_cost = costs[1][0] if len(costs) > 1 else best_cost
                regret = second_best_cost - best_cost
                if regret > best_regret:
                    best_regret = regret
                    best_cust = cust
                    best_r = costs[0][1]
                    best_pos = costs[0][2]
            # Insert best_cust at best position
            routes[best_r].insert(best_pos, best_cust)
            route_dists[best_r] = route_distance(routes[best_r])
            unassigned.remove(best_cust)
        return routes, route_dists

    def local_search(routes, route_dists):
        best_routes = [list(r) for r in routes]
        best_max = max(route_dists)
        try:
            report_best_vrp(best_routes)
        except NameError:
            pass
        max_iters = 20  # number of full passes for each operator
        for iteration in range(max_iters):
            improved = False
            # Relocate
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 2:
                    continue
                for idx in range(1, len(route1)-1):
                    cust = route1[idx]
                    new_route1 = route1[:idx] + route1[idx+1:]
                    new_dist1 = route_distance(new_route1)
                    for r2 in range(truck_count):
                        if r2 == r1:
                            continue
                        route2 = routes[r2]
                        for pos in range(1, len(route2)):
                            inc = compute_insertion_cost(route2, pos, cust)
                            new_dist2 = route_dists[r2] + inc
                            other_dists = [route_dists[i] for i in range(truck_count) if i not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < best_max - 1e-9:
                                routes[r1] = new_route1
                                route_dists[r1] = new_dist1
                                routes[r2].insert(pos, cust)
                                route_dists[r2] = new_dist2
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                                improved = True
                                try:
                                    report_best_vrp(best_routes)
                                except NameError:
                                    pass
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Swap
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 2:
                    continue
                for idx1 in range(1, len(route1)-1):
                    cust1 = route1[idx1]
                    for r2 in range(r1+1, truck_count):
                        route2 = routes[r2]
                        if len(route2) <= 2:
                            continue
                        for idx2 in range(1, len(route2)-1):
                            cust2 = route2[idx2]
                            new_route1 = route1[:idx1] + [cust2] + route1[idx1+1:]
                            new_route2 = route2[:idx2] + [cust1] + route2[idx2+1:]
                            new_dist1 = route_distance(new_route1)
                            new_dist2 = route_distance(new_route2)
                            other_dists = [route_dists[i] for i in range(truck_count) if i not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < best_max - 1e-9:
                                routes[r1] = new_route1
                                routes[r2] = new_route2
                                route_dists[r1] = new_dist1
                                route_dists[r2] = new_dist2
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                                improved = True
                                try:
                                    report_best_vrp(best_routes)
                                except NameError:
                                    pass
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Intra-route 2-opt
            for r in range(truck_count):
                route = routes[r]
                if len(route) <= 3:
                    continue
                best_improve = 0.0
                best_i = best_j = -1
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < route_dists[r] - 1e-9:
                            improvement = route_dists[r] - new_dist
                            if improvement > best_improve:
                                best_improve = improvement
                                best_i, best_j = i, j
                if best_improve > 0:
                    new_route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                    routes[r] = new_route
                    route_dists[r] = route_distance(new_route)
                    new_max = max(route_dists)
                    if new_max < best_max - 1e-9:
                        best_max = new_max
                        best_routes = [list(r) for r in routes]
                        try:
                            report_best_vrp(best_routes)
                        except NameError:
                            pass
                    improved = True
                    break
            if improved:
                continue
            # Cross-route 2-opt
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 2:
                    continue
                for r2 in range(r1+1, truck_count):
                    route2 = routes[r2]
                    if len(route2) <= 2:
                        continue
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i+1] + route2[j+1:]
                            new2 = route2[:j+1] + route1[i+1:]
                            new_dist1 = route_distance(new1)
                            new_dist2 = route_distance(new2)
                            other_dists = [route_dists[k] for k in range(truck_count) if k not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < best_max - 1e-9:
                                routes[r1] = new1
                                routes[r2] = new2
                                route_dists[r1] = new_dist1
                                route_dists[r2] = new_dist2
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                                improved = True
                                try:
                                    report_best_vrp(best_routes)
                                except NameError:
                                    pass
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return best_routes, best_max

    def perturb(routes, route_dists):
        # Randomly relocate a small number of customers (5% of n) to different routes
        n_cust = n - 1
        perturb_count = max(1, int(n_cust * 0.05))
        customers = list(range(1, n))
        random.shuffle(customers)
        for cust in customers[:perturb_count]:
            # Find current route of this customer
            for r in range(truck_count):
                if cust in routes[r]:
                    idx = routes[r].index(cust)
                    # Remove customer
                    new_route = routes[r][:idx] + routes[r][idx+1:]
                    new_dist_r = route_distance(new_route)
                    # Find best insertion in another route (avoid same truck to reduce chance of trivial move)
                    best_new_max = float('inf')
                    best_r_target = None
                    best_pos_target = None
                    for r2 in range(truck_count):
                        if r2 == r:
                            continue
                        route2 = routes[r2]
                        for pos in range(1, len(route2)):
                            inc = compute_insertion_cost(route2, pos, cust)
                            new_dist2 = route_dists[r2] + inc
                            other_dists = [route_dists[i] for i in range(truck_count) if i not in (r, r2)]
                            new_max = max(new_dist_r, new_dist2, *other_dists)
                            if new_max < best_new_max - 1e-9:
                                best_new_max = new_max
                                best_r_target = r2
                                best_pos_target = pos
                    if best_r_target is not None:
                        # Apply move
                        routes[r] = new_route
                        route_dists[r] = new_dist_r
                        routes[best_r_target].insert(best_pos_target, cust)
                        route_dists[best_r_target] = route_distance(routes[best_r_target])
        return routes, route_dists

    # Main
    num_restarts = 20
    best_routes = []
    best_max = float('inf')
    for restart in range(num_restarts):
        # Adaptive lambda: start with 1.0, adjust based on imbalance?
        # Simple: use lambda = 1.0 + (max_route - avg_route)/avg_route if avg>0
        # But we don't know yet; we can compute after construction?
        # We'll construct with lambda=1.0 first, then adjust after local search by observing the result?
        # For simplicity, use fixed lambda=1.0 for initial construction; we can adapt later if needed.
        lambda_weight = 1.0
        routes, route_dists = construct_regret_balancing(lambda_weight)
        routes, max_dist = local_search(routes, route_dists)
        # Perturbation only if not best? Always perturb to explore diversity.
        routes_pert, dists_pert = perturb(routes, route_dists)
        # Run local search again on perturbed solution
        routes_pert, max_dist_pert = local_search(routes_pert, dists_pert)
        if max_dist_pert < max_dist - 1e-9:
            routes, max_dist = routes_pert, max_dist_pert
        # Update best
        if max_dist < best_max - 1e-9:
            best_routes = [list(r) for r in routes]
            best_max = max_dist
            try:
                report_best_vrp(best_routes)
            except NameError:
                pass
    # Ensure empty routes
    for r in range(truck_count):
        if len(best_routes[r]) == 0:
            best_routes[r] = [0, 0]
    return best_routes