import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0 for _ in range(truck_count)]

    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def insert_cost(route, customer, pos):
        return (distance_matrix[route[pos-1], customer] + distance_matrix[customer, route[pos]] - distance_matrix[route[pos-1], route[pos]])

    # Construction: regret-based insertion
    customers = list(range(1, n))
    while customers:
        best_cust = None
        best_regret = -1.0
        best_route = -1
        best_pos = -1
        best_max = float('inf')
        for cust in customers:
            costs = []
            for r in range(truck_count):
                route = routes[r]
                for pos in range(1, len(route)):
                    inc = insert_cost(route, cust, pos)
                    new_dist = route_dists[r] + inc
                    new_max = max(new_dist, max(route_dists[:r] + route_dists[r+1:]))
                    costs.append((new_max, r, pos))
            costs.sort(key=lambda x: x[0])
            if len(costs) >= 2:
                regret = costs[1][0] - costs[0][0]
            else:
                regret = costs[0][0]
            if regret > best_regret:
                best_regret = regret
                best_cust = cust
                best_max = costs[0][0]
                best_route = costs[0][1]
                best_pos = costs[0][2]
        routes[best_route].insert(best_pos, best_cust)
        route_dists[best_route] = route_distance(routes[best_route])
        customers.remove(best_cust)

    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    # Local search and restart loop (initial + 5 restarts)
    for restart in range(6):  # 0..5, break after 5
        # Local search improvement
        max_iters = 10 * n * truck_count
        for _ in range(max_iters):
            improved = False
            # Relocate moves
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
                            inc = insert_cost(route2, cust, pos)
                            new_dist2 = route_dists[r2] + inc
                            other_dists = [route_dists[i] for i in range(truck_count) if i not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < best_max:
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

            # Swap moves
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
                            if new_max < best_max:
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
                    if new_max < best_max:
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
                            if new_max < best_max:
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

        if restart >= 5:  # after 5th restart, no more ruin
            break

        # Restart: targeted ruin: remove 30% of customers from the route with highest max distance
        # Find route with max distance
        max_route_idx = max(range(truck_count), key=lambda i: route_dists[i])
        target_route = routes[max_route_idx]
        if len(target_route) <= 2:  # no customers to remove
            continue
        # Number of customers to remove: 30% of customers in that route (exclude depots)
        num_cust_in_route = len(target_route) - 2
        num_remove = max(1, int(0.3 * num_cust_in_route))
        # Select random customers from target route (positions 1..-2)
        candidates = list(range(1, len(target_route)-1))
        remove_indices = random.sample(candidates, min(num_remove, len(candidates)))
        remove_customers = [target_route[i] for i in remove_indices]
        # Remove them
        for cust in remove_customers:
            # Find and remove from whichever route it's in (should be target_route)
            for r in range(truck_count):
                route = routes[r]
                if cust in route:
                    idx = route.index(cust)
                    route.pop(idx)
                    break
        # Update route distances
        for r in range(truck_count):
            route_dists[r] = route_distance(routes[r])

        # Reinsert removed customers using regret-based insertion
        customers = remove_customers[:]
        random.shuffle(customers)
        while customers:
            best_cust = None
            best_regret = -1.0
            best_route = -1
            best_pos = -1
            best_max = float('inf')
            for cust in customers:
                costs = []
                for r in range(truck_count):
                    route = routes[r]
                    for pos in range(1, len(route)):
                        inc = insert_cost(route, cust, pos)
                        new_dist = route_dists[r] + inc
                        new_max = max(new_dist, max(route_dists[:r] + route_dists[r+1:]))
                        costs.append((new_max, r, pos))
                costs.sort(key=lambda x: x[0])
                if len(costs) >= 2:
                    regret = costs[1][0] - costs[0][0]
                else:
                    regret = costs[0][0]
                if regret > best_regret:
                    best_regret = regret
                    best_cust = cust
                    best_max = costs[0][0]
                    best_route = costs[0][1]
                    best_pos = costs[0][2]
            routes[best_route].insert(best_pos, best_cust)
            route_dists[best_route] = route_distance(routes[best_route])
            customers.remove(best_cust)

        # Update best if improved
        cur_max = max(route_dists)
        if cur_max < best_max:
            best_max = cur_max
            best_routes = [list(r) for r in routes]
            try:
                report_best_vrp(best_routes)
            except NameError:
                pass

    # Ensure empty routes
    for r in range(truck_count):
        if len(best_routes[r]) == 0:
            best_routes[r] = [0, 0]
    return best_routes