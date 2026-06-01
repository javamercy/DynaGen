import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    # Initialize empty routes
    routes = [[0, 0] for _ in range(truck_count)]
    
    # List of unassigned customers (nodes 1..n-1)
    unassigned = list(range(1, n))
    
    # Helper to compute cost of a route (total distance)
    def route_cost(route):
        cost = 0.0
        for i in range(len(route)-1):
            cost += distance_matrix[route[i], route[i+1]]
        return cost
    
    # Best insertion position and cost for a customer in a given route
    def best_insertion(customer, route):
        best_cost = float('inf')
        best_pos = 1  # position after start depot (0) and before end depot (0)
        # route always has at least [0,0] so len>=2
        for pos in range(1, len(route)):
            # cost increase by inserting between route[pos-1] and route[pos]
            delta = (distance_matrix[route[pos-1], customer] +
                     distance_matrix[customer, route[pos]] -
                     distance_matrix[route[pos-1], route[pos]])
            if delta < best_cost:
                best_cost = delta
                best_pos = pos
        return best_pos, best_cost
    
    # Regret insertion
    while unassigned:
        best_customer = None
        best_route_idx = None
        best_position = None
        max_regret = -float('inf')
        
        for cust in unassigned:
            costs = []
            positions = []
            for r_idx, route in enumerate(routes):
                pos, cost = best_insertion(cust, route)
                costs.append(cost)
                positions.append(pos)
            # Sort costs to find best and second best
            sorted_costs = sorted(costs)
            if len(sorted_costs) >= 2:
                regret = sorted_costs[1] - sorted_costs[0]
            else:
                regret = sorted_costs[0]  # only one route
            if regret > max_regret:
                max_regret = regret
                best_customer = cust
                best_route_idx = int(np.argmin(costs))
                best_position = positions[best_route_idx]
            elif regret == max_regret and best_customer is not None and cust < best_customer:
                # tie-break by smaller node index
                best_customer = cust
                best_route_idx = int(np.argmin(costs))
                best_position = positions[best_route_idx]
        
        # Insert best_customer into best_route at best_position
        route = routes[best_route_idx]
        route.insert(best_position, best_customer)
        unassigned.remove(best_customer)
    
    # Compute initial max route distance
    costs = [route_cost(r) for r in routes]
    best_max = max(costs)
    
    # Improvement: intra-route 2-opt and inter-route relocate
    improved = True
    max_passes = 2  # bounded
    for _ in range(max_passes):
        if not improved:
            break
        improved = False
        # Intra-route 2-opt for each route
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            # bounded by route length^2
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # reverse segment [i, j]
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_cost = route_cost(new_route)
                    if new_cost < route_cost(route):
                        routes[r_idx] = new_route
                        improved = True
        # Update costs after intra-route
        costs = [route_cost(r) for r in routes]
        new_max = max(costs)
        if new_max < best_max:
            best_max = new_max
            # report_best_vrp(routes)  # assume available
        
        # Inter-route relocate: try moving each customer to another route
        for r_idx in range(truck_count):
            route = routes[r_idx]
            # iterate over customers (skip depot)
            for pos in range(1, len(route)-1):
                cust = route[pos]
                # remove customer temporarily
                temp_route = route[:pos] + route[pos+1:]
                # find best insertion in other routes
                best_other_cost = float('inf')
                best_other_route = None
                best_other_pos = None
                for other_idx, other_route in enumerate(routes):
                    if other_idx == r_idx:
                        continue
                    o_pos, o_cost = best_insertion(cust, other_route)
                    if o_cost < best_other_cost:
                        best_other_cost = o_cost
                        best_other_route = other_idx
                        best_other_pos = o_pos
                # compute potential new max if we move
                # current max
                old_max = max(costs)
                # costs after move: we need to compute new costs for both routes
                # original route cost (without cust)
                old_route_cost = route_cost(route)
                old_other_cost = costs[best_other_route]
                # new route cost after removal: route_cost(temp_route)
                new_route_cost = route_cost(temp_route)
                # new other route cost after insertion: insert cust
                new_other_route = routes[best_other_route].copy()
                new_other_route.insert(best_other_pos, cust)
                new_other_cost = route_cost(new_other_route)
                new_max = max(old_max, new_route_cost, new_other_cost)
                if new_max < old_max:
                    # commit move
                    routes[r_idx] = temp_route
                    routes[best_other_route] = new_other_route
                    improved = True
                    costs = [route_cost(r) for r in routes]
                    break  # restart outer loop? but keep simple
            if improved:
                break
        # Update best if max decreased
        costs = [route_cost(r) for r in routes]
        current_max = max(costs)
        if current_max < best_max:
            best_max = current_max
            # report_best_vrp(routes)
    
    return routes