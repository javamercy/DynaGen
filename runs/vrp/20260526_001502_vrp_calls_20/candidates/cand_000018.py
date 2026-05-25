import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)  # deterministic random
    n = distance_matrix.shape[0]
    # initialize routes
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
        return (distance_matrix[route[pos-1], customer] + distance_matrix[customer, route[pos]] -
                distance_matrix[route[pos-1], route[pos]])
    
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
        # Insert best_cust
        routes[best_route].insert(best_pos, best_cust)
        route_dists[best_route] = route_distance(routes[best_route])
        customers.remove(best_cust)
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass
    
    # Local search and restart loop
    for restart in range(4):  # one initial + up to 3 restarts
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
        if restart >= 3:  # after final restart, no more
            break
        # Restart: ruin and recreate
        # Remove 20% of customers randomly
        all_customers = list(range(1, n))
        num_remove = max(1, int(0.2 * (n-1)))
        selected = random.sample(all_customers, num_remove)
        # Remove these customers from routes
        for cust in selected:
            for r in range(truck_count):
                route = routes[r]
                if cust in route:
                    idx = route.index(cust)
                    route.pop(idx)
                    break
        # Update route distances
        for r in range(truck_count):
            route_dists[r] = route_distance(routes[r])
        # Reinsert selected customers using regret-based insertion
        customers = selected[:]
        random.shuffle(customers)  # deterministic? shuffle with random
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