import numpy as np
from collections import defaultdict

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    # Initialize all routes with depot
    routes = [[0, 0] for _ in range(truck_count)]
    # distances per route
    dists = [0.0] * truck_count
    # remaining customers
    customers = list(range(1, n))
    
    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    def eval_route_dist(route):
        return route_distance(route)
    
    def compute_insertion_cost(route, pos, cust):
        # cost of inserting cust at position pos in route (0 < pos < len(route))
        prev = route[pos-1]
        next = route[pos]
        return distance_matrix[prev, cust] + distance_matrix[cust, next] - distance_matrix[prev, next]
    
    # Regret-2 construction
    for _ in range(len(customers)):
        # For each unassigned customer, compute for each route the best insertion cost and position
        best_cust = None
        best_regret = -float('inf')
        best_truck = None
        best_pos = None
        best_increase = float('inf')
        for cust in customers:
            # collect best and second best insertion improvements (increase in max distance)
            improvements = []
            for t in range(truck_count):
                route = routes[t]
                current_dist = dists[t]
                # consider all insertion positions from 1 to len(route)-1 (inclusive)
                for pos in range(1, len(route)):
                    new_dist = current_dist + compute_insertion_cost(route, pos, cust)
                    # new max distance if we insert
                    other_dists = [dists[i] for i in range(truck_count) if i != t]
                    new_max = max(new_dist, max(other_dists) if other_dists else 0.0)
                    improvement = -(new_max - max(dists))  # negative: larger reduction better (but we want min increase)
                    improvements.append((improvement, new_max, t, pos, cust))
            # improvements sorted by improvement (smaller is better), then by new_max, then truck, then pos, then cust
            improvements.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
            if improvements:
                best_improve = improvements[0][0]
                best_new_max = improvements[0][1]
                if len(improvements) > 1:
                    second_improve = improvements[1][0]
                else:
                    second_improve = best_improve  # no second
                regret = second_improve - best_improve
                # higher regret means more urgent
                # Tie-breaking: larger regret, then smaller customer index
                if (regret > best_regret) or (regret == best_regret and (best_cust is None or cust < best_cust)):
                    best_regret = regret
                    best_cust = cust
                    # store the best insertion details (from the first improvement)
                    best_truck = improvements[0][2]
                    best_pos = improvements[0][3]
                    best_new_max_used = best_new_max
        # Insert best_cust at best_truck, best_pos
        route = routes[best_truck]
        route.insert(best_pos, best_cust)
        dists[best_truck] = route_distance(route)
        # Update current max
        current_max = max(dists)
        # Report best so far (maybe not best, but we will report after construction)
        customers.remove(best_cust)
    
    # After construction, report initial solution
    best_routes = [list(r) for r in routes]
    best_max = max(dists)
    report_best_vrp(best_routes)
    
    # Helper to evaluate max distance if a change is made
    def eval_modified(modified_routes, modified_dists):
        return max(modified_dists)
    
    # Local search: combine 2-opt*, relocate, swap
    n_customers = n-1
    # Main loop: iterate until no improvement or max iterations
    max_iterations = n_customers * truck_count  # bound
    iteration = 0
    improved = True
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        # 2-opt*: exchange edges from two different routes
        # For each pair of routes (t1, t2) with t1 < t2
        for t1 in range(truck_count):
            for t2 in range(t1+1, truck_count):
                route1 = routes[t1]
                route2 = routes[t2]
                # Only consider inner segments: from 1 to len-2 (excluding depot)
                len1 = len(route1)
                len2 = len(route2)
                # i1, i2 are the start of the edge to break (from i to i+1)
                for i1 in range(1, len1-2):
                    for i2 in range(1, len2-2):
                        # new routes after swapping tails
                        new_route1 = route1[:i1+1] + route2[i2+1:]
                        new_route2 = route2[:i2+1] + route1[i1+1:]
                        # Check feasibility (depots at ends)
                        if new_route1[0] != 0 or new_route1[-1] != 0 or new_route2[0] != 0 or new_route2[-1] != 0:
                            continue
                        new_dist1 = route_distance(new_route1)
                        new_dist2 = route_distance(new_route2)
                        other_dists = [dists[i] for i in range(truck_count) if i != t1 and i != t2]
                        new_max = max(new_dist1, new_dist2, max(other_dists) if other_dists else 0.0)
                        if new_max < best_max:
                            # Accept
                            routes[t1] = new_route1
                            routes[t2] = new_route2
                            dists[t1] = new_dist1
                            dists[t2] = new_dist2
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Relocate: move a customer from one route to another
        for src in range(truck_count):
            route_src = routes[src]
            # iterate over positions to remove (excluding depots)
            for pos_remove in range(1, len(route_src)-1):
                cust = route_src[pos_remove]
                temp_src = route_src[:pos_remove] + route_src[pos_remove+1:]
                new_dist_src = route_distance(temp_src)
                # try inserting into other routes
                best_dst = None
                best_pos_insert = None
                best_new_max_reloc = float('inf')
                for dst in range(truck_count):
                    if dst == src:
                        continue
                    route_dst = routes[dst]
                    for pos_insert in range(1, len(route_dst)):
                        new_route_dst = route_dst[:pos_insert] + [cust] + route_dst[pos_insert:]
                        new_dist_dst = route_distance(new_route_dst)
                        other_dists = [dists[i] for i in range(truck_count) if i != src and i != dst]
                        new_max = max(new_dist_src, new_dist_dst, max(other_dists) if other_dists else 0.0)
                        if new_max < best_max:
                            best_dst = dst
                            best_pos_insert = pos_insert
                            best_new_max_reloc = new_max
                if best_dst is not None:
                    # Apply move
                    routes[src] = temp_src
                    dists[src] = new_dist_src
                    routes[best_dst] = routes[best_dst][:best_pos_insert] + [cust] + routes[best_dst][best_pos_insert:]
                    dists[best_dst] = route_distance(routes[best_dst])
                    best_max = best_new_max_reloc
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # Swap: exchange two customers from different routes
        for t1 in range(truck_count):
            for t2 in range(t1+1, truck_count):
                route1 = routes[t1]
                route2 = routes[t2]
                # iterate over customer positions (not depots)
                for p1 in range(1, len(route1)-1):
                    for p2 in range(1, len(route2)-1):
                        cust1 = route1[p1]
                        cust2 = route2[p2]
                        # swap
                        new_route1 = route1[:p1] + [cust2] + route1[p1+1:]
                        new_route2 = route2[:p2] + [cust1] + route2[p2+1:]
                        new_dist1 = route_distance(new_route1)
                        new_dist2 = route_distance(new_route2)
                        other_dists = [dists[i] for i in range(truck_count) if i != t1 and i != t2]
                        new_max = max(new_dist1, new_dist2, max(other_dists) if other_dists else 0.0)
                        if new_max < best_max:
                            routes[t1] = new_route1
                            routes[t2] = new_route2
                            dists[t1] = new_dist1
                            dists[t2] = new_dist2
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    # Return best found
    return best_routes