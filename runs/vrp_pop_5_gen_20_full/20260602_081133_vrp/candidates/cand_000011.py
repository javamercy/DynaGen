import numpy as np
import heapq

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    # Precompute distance from depot to each node
    depot = 0
    # Initialize routes: each starts and ends at depot
    routes = [[0, 0] for _ in range(truck_count)]
    assigned = [False] * n
    assigned[0] = True
    unassigned = list(range(1, n))
    
    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        total = distance_matrix[route[-1]][route[0]]  # closing to depot
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total
    
    def insertion_cost(route, cust):
        """cost increase of inserting cust into best position in route"""
        best = float('inf')
        for pos in range(1, len(route)):
            cost = (distance_matrix[route[pos-1]][cust] +
                    distance_matrix[cust][route[pos]] -
                    distance_matrix[route[pos-1]][route[pos]])
            if cost < best:
                best = cost
        return best
    
    def best_insert(route, cust):
        """return (cost, position) for best insertion"""
        best = float('inf')
        best_pos = 1
        for pos in range(1, len(route)):
            cost = (distance_matrix[route[pos-1]][cust] +
                    distance_matrix[cust][route[pos]] -
                    distance_matrix[route[pos-1]][route[pos]])
            if cost < best - 1e-12:
                best = cost
                best_pos = pos
        return best, best_pos
    
    # Construction: regret heuristic
    while unassigned:
        # for each unassigned customer, compute best and second best insertion cost
        regrets = []
        for cust in unassigned:
            costs = []
            for t in range(truck_count):
                cost, _ = best_insert(routes[t], cust)
                costs.append(cost)
            # get two smallest
            smallest = heapq.nsmallest(2, costs)
            regret = smallest[1] - smallest[0] if len(smallest) > 1 else smallest[0]
            # tie-breaker: larger regret, then smaller customer index
            regrets.append((-regret, cust))
        heapq.heapify(regrets)
        # select customer with max regret (negative min)
        _, cust = heapq.heappop(regrets)
        # find best truck and position
        best_truck = 0
        best_pos = 1
        best_cost = float('inf')
        for t in range(truck_count):
            cost, pos = best_insert(routes[t], cust)
            if cost < best_cost - 1e-12:
                best_cost = cost
                best_truck = t
                best_pos = pos
        # insert
        routes[best_truck].insert(best_pos, cust)
        unassigned.remove(cust)
    
    # Compute distances
    dists = [route_distance(r) for r in routes]
    max_dist = max(dists)
    try:
        report_best_vrp(routes)
    except:
        pass
    
    # Improvement: bounded iterations
    max_iter = n * truck_count * 5
    for _ in range(max_iter):
        improved = False
        # Intra-route 2-opt
        for t in range(truck_count):
            route = routes[t]
            if len(route) <= 3:
                continue
            best_dist = dists[t]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist - 1e-12:
                        routes[t] = new_route
                        best_dist = new_dist
                        improved = True
        if improved:
            dists = [route_distance(r) for r in routes]
            new_max = max(dists)
            if new_max < max_dist - 1e-12:
                max_dist = new_max
                try:
                    report_best_vrp(routes)
                except:
                    pass
            continue
        
        # Inter-route swap
        best_swap = None
        best_reduction = 0.0
        for t1 in range(truck_count):
            for t2 in range(t1+1, truck_count):
                route1 = routes[t1]
                route2 = routes[t2]
                # Skip if routes are just depot-depot
                if len(route1) <= 2 and len(route2) <= 2:
                    continue
                for i in range(1, len(route1)-1):
                    for j in range(1, len(route2)-1):
                        cust1 = route1[i]
                        cust2 = route2[j]
                        new_r1 = route1[:i] + [cust2] + route1[i+1:]
                        new_r2 = route2[:j] + [cust1] + route2[j+1:]
                        d1 = route_distance(new_r1)
                        d2 = route_distance(new_r2)
                        new_max = max(d1, d2, max(dists[:t1]), max(dists[t1+1:t2]), max(dists[t2+1:]))
                        reduction = max_dist - new_max
                        if reduction > best_reduction + 1e-12:
                            best_reduction = reduction
                            best_swap = (t1, t2, i, j, new_r1, new_r2, d1, d2)
        if best_reduction > 1e-12:
            t1, t2, i, j, new_r1, new_r2, d1, d2 = best_swap
            routes[t1] = new_r1
            routes[t2] = new_r2
            dists[t1] = d1
            dists[t2] = d2
            max_dist = max(dists)
            try:
                report_best_vrp(routes)
            except:
                pass
            improved = True
            continue
        
        # Inter-route relocate from longest route
        max_idx = max(range(truck_count), key=lambda t: dists[t])
        max_dist = dists[max_idx]
        best_move = None
        best_red = 0.0
        route_long = routes[max_idx]
        for i in range(1, len(route_long)-1):
            cust = route_long[i]
            new_long = route_long[:i] + route_long[i+1:]
            d_long = route_distance(new_long)
            for t in range(truck_count):
                if t == max_idx:
                    continue
                # best insertion in route t
                best_cost = float('inf')
                best_pos = 1
                for pos in range(1, len(routes[t])):
                    cost = (distance_matrix[routes[t][pos-1]][cust] +
                            distance_matrix[cust][routes[t][pos]] -
                            distance_matrix[routes[t][pos-1]][routes[t][pos]])
                    if cost < best_cost - 1e-12:
                        best_cost = cost
                        best_pos = pos
                new_other = routes[t][:best_pos] + [cust] + routes[t][best_pos:]
                d_other = route_distance(new_other)
                other_dists = [dists[i] for i in range(truck_count) if i not in (max_idx, t)]
                new_max = max(d_long, d_other, *other_dists)
                reduction = max_dist - new_max
                if reduction > best_red + 1e-12:
                    best_red = reduction
                    best_move = (max_idx, t, new_long, new_other, d_long, d_other)
        if best_red > 1e-12 and best_move is not None:
            max_idx, t, new_long, new_other, d_long, d_other = best_move
            routes[max_idx] = new_long
            routes[t] = new_other
            dists[max_idx] = d_long
            dists[t] = d_other
            max_dist = max(dists)
            try:
                report_best_vrp(routes)
            except:
                pass
            improved = True
            continue
        
        # No improvement
        break
    
    return routes