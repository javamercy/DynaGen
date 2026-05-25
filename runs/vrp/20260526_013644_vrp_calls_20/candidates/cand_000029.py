import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    unassigned = set(customers)
    routes = [[depot, depot] for _ in range(truck_count)]

    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def best_insertion(cust, route):
        best_cost = float('inf')
        best_pos = -1
        for pos in range(1, len(route)):
            i = route[pos-1]
            j = route[pos]
            cost = distance_matrix[i, cust] + distance_matrix[cust, j] - distance_matrix[i, j]
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        return best_cost, best_pos

    # Regret insertion construction
    while unassigned:
        best_regret = -1.0
        best_cust = -1
        best_route_idx = -1
        best_pos = -1
        best_cost_for_cust = float('inf')
        for cust in list(unassigned):
            costs = []
            for r_idx, route in enumerate(routes):
                cost, pos = best_insertion(cust, route)
                costs.append((cost, r_idx, pos))
            costs.sort(key=lambda x: x[0])
            if len(costs) == 1:
                regret = 1e9
            else:
                regret = costs[1][0] - costs[0][0]
            if (regret > best_regret or
                (regret == best_regret and costs[0][0] > best_cost_for_cust) or
                (regret == best_regret and costs[0][0] == best_cost_for_cust and cust < best_cust)):
                best_regret = regret
                best_cust = cust
                best_cost_for_cust = costs[0][0]
                best_route_idx = costs[0][1]
                best_pos = costs[0][2]
        route = routes[best_route_idx]
        route.insert(best_pos, best_cust)
        unassigned.remove(best_cust)

    best_routes = [list(r) for r in routes]
    best_max = max(route_dist(r) for r in best_routes)
    report_best_vrp(best_routes)

    n_cust = n - 1
    max_outer_iters = 2 * n_cust
    max_perturbs = n_cust // 2
    for outer_iter in range(max_outer_iters):
        improved = False
        # Inter-route relocate: best improvement
        best_move = None
        best_new_max = best_max
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for cust in route[1:-1]:
                new_route = [x for x in route if x != cust]
                for other_idx, other_route in enumerate(routes):
                    if other_idx == r_idx:
                        continue
                    cost, pos = best_insertion(cust, other_route)
                    candidate_routes = [list(r) for r in routes]
                    candidate_routes[r_idx] = new_route
                    other_new = list(other_route)
                    other_new.insert(pos, cust)
                    candidate_routes[other_idx] = other_new
                    dists = [route_dist(r) for r in candidate_routes]
                    new_max = max(dists)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_move = (r_idx, cust, other_idx, pos, candidate_routes)
        if best_move is not None:
            routes = best_move[4]
            best_max = best_new_max
            improved = True
            report_best_vrp(routes)

        # Intra-route 2-opt
        if not improved:
            for r_idx, route in enumerate(routes):
                if len(route) <= 4:
                    continue
                best_imp = None
                best_dist = route_dist(route)
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_dist(new_route)
                        if new_dist < best_dist:
                            best_dist = new_dist
                            best_imp = (i, j, new_route)
                if best_imp is not None:
                    routes[r_idx] = best_imp[2]
                    improved = True
                    dists = [route_dist(r) for r in routes]
                    new_max = max(dists)
                    if new_max < best_max:
                        best_max = new_max
                        report_best_vrp(routes)

        # Cross-route 2-opt*
        if not improved:
            best_move = None
            best_new_max = best_max
            for i in range(len(routes)):
                for j in range(i+1, len(routes)):
                    r1 = routes[i]
                    r2 = routes[j]
                    if len(r1) <= 2 or len(r2) <= 2:
                        continue
                    for i1 in range(1, len(r1)-1):
                        for j1 in range(1, len(r2)-1):
                            new_r1 = r1[:i1] + r2[j1:]
                            new_r2 = r2[:j1] + r1[i1:]
                            candidate_routes = [list(r) for r in routes]
                            candidate_routes[i] = new_r1
                            candidate_routes[j] = new_r2
                            dists = [route_dist(r) for r in candidate_routes]
                            new_max = max(dists)
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = (i, j, i1, j1, new_r1, new_r2)
            if best_move is not None:
                i, j, i1, j1, new_r1, new_r2 = best_move
                routes[i] = new_r1
                routes[j] = new_r2
                best_max = best_new_max
                improved = True
                report_best_vrp(routes)

        # Perturbation if no improvement
        if not improved:
            # Count perturbations performed in this outer loop
            if outer_iter >= max_perturbs:
                break
            # Deterministic perturbation: move one customer from longest route to another
            max_dist = -1
            longest_idx = -1
            for r_idx, route in enumerate(routes):
                d = route_dist(route)
                if d > max_dist:
                    max_dist = d
                    longest_idx = r_idx
            # Find customer in longest route whose removal reduces distance most
            route_long = routes[longest_idx]
            if len(route_long) <= 3:
                break
            best_cust = None
            best_red = -1.0
            for cust in route_long[1:-1]:
                new_route = [x for x in route_long if x != cust]
                new_dist = route_dist(new_route)
                red = max_dist - new_dist
                if red > best_red:
                    best_red = red
                    best_cust = cust
            if best_cust is None:
                break
            # Insert best_cust into another route (deterministic: the one that minimizes new max)
            best_other = None
            best_pos = -1
            best_new_max = best_max
            for other_idx, other_route in enumerate(routes):
                if other_idx == longest_idx:
                    continue
                cost, pos = best_insertion(best_cust, other_route)
                candidate_routes = [list(r) for r in routes]
                # Remove from longest
                candidate_routes[longest_idx] = [x for x in route_long if x != best_cust]
                # Insert into other
                other_new = list(other_route)
                other_new.insert(pos, best_cust)
                candidate_routes[other_idx] = other_new
                dists = [route_dist(r) for r in candidate_routes]
                new_max = max(dists)
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_other = other_idx
                    best_pos = pos
            if best_other is not None:
                # Apply move
                routes[longest_idx] = [x for x in route_long if x != best_cust]
                other_new = list(routes[best_other])
                other_new.insert(best_pos, best_cust)
                routes[best_other] = other_new
                # Update best_max if better
                dists = [route_dist(r) for r in routes]
                new_max = max(dists)
                if new_max < best_max:
                    best_max = new_max
                    report_best_vrp(routes)
                improved = True
            else:
                break

        if not improved:
            break

    # Ensure exactly truck_count routes with depots
    result = []
    for r in routes:
        if len(r) <= 2:
            result.append([0, 0])
        else:
            if r[0] != 0:
                r.insert(0, 0)
            if r[-1] != 0:
                r.append(0)
            result.append(r)
    while len(result) < truck_count:
        result.append([0, 0])
    return result