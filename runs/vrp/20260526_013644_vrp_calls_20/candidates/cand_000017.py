import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    unassigned = set(customers)
    routes = [[depot, depot] for _ in range(truck_count)]
    
    def route_dist(route):
        d = 0
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
    
    # Construction
    while unassigned:
        best_regret = -1
        best_cust = -1
        best_route_idx = -1
        best_pos = -1
        best_cost_for_cust = float('inf')
        
        for cust in unassigned:
            costs = []
            for r_idx, route in enumerate(routes):
                cost, pos = best_insertion(cust, route)
                costs.append((cost, r_idx, pos))
            costs.sort(key=lambda x: x[0])
            if len(costs) == 1:
                regret = costs[0][0] * 2
            else:
                regret = costs[1][0] - costs[0][0]
            if regret > best_regret or (regret == best_regret and costs[0][0] > best_cost_for_cust):
                best_regret = regret
                best_cust = cust
                best_cost_for_cust = costs[0][0]
                best_route_idx = costs[0][1]
                best_pos = costs[0][2]
            elif regret == best_regret and costs[0][0] == best_cost_for_cust:
                if cust < best_cust:
                    best_cust = cust
                    best_route_idx = costs[0][1]
                    best_pos = costs[0][2]
        route = routes[best_route_idx]
        route.insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    
    # Initial solution
    best_routes = [list(r) for r in routes]
    best_max = max(route_dist(r) for r in best_routes)
    report_best_vrp(best_routes)  # assumed defined elsewhere
    
    # Precompute route distances
    dists = [route_dist(r) for r in routes]
    
    # Helper to evaluate max after modifying two routes (inter-route relocate)
    def evaluate_relocate(cust, from_route, to_route, to_pos):
        new_from = [x for x in from_route if x != cust]
        new_to = list(to_route)
        new_to.insert(to_pos, cust)
        new_dists = dists.copy()
        new_dists[from_idx] = route_dist(new_from)
        new_dists[to_idx] = route_dist(new_to)
        new_max = max(new_dists)
        return new_max, new_from, new_to
    
    max_iters = 3 * n
    for iteration in range(max_iters):
        improved = False
        # Find longest route
        max_dist = max(dists)
        longest_idx = dists.index(max_dist)
        longest_route = routes[longest_idx]
        if len(longest_route) <= 3:
            # no interior customers
            break
        # Inter-route relocate: try moving each customer of longest route to other routes
        for cust in longest_route[1:-1]:
            # remove cust from longest route
            temp_route = [x for x in longest_route if x != cust]
            for other_idx, other_route in enumerate(routes):
                if other_idx == longest_idx:
                    continue
                # best insertion in other route
                cost, pos = best_insertion(cust, other_route)
                new_max, new_from, new_to = evaluate_relocate(cust, longest_route, other_route, pos)
                if new_max < best_max:
                    best_max = new_max
                    routes[longest_idx] = new_from
                    routes[other_idx] = new_to
                    dists[longest_idx] = route_dist(new_from)
                    dists[other_idx] = route_dist(new_to)
                    improved = True
                    report_best_vrp([list(r) for r in routes])
                    break
            if improved:
                break
        if not improved:
            # Intra-route 2-opt on longest route
            longest_route = routes[longest_idx]
            n_nodes = len(longest_route)
            best_imp = False
            for i in range(1, n_nodes-2):
                for j in range(i+1, n_nodes-1):
                    new_route = longest_route[:i] + longest_route[i:j+1][::-1] + longest_route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < dists[longest_idx]:
                        new_max = max(max(dists[:longest_idx] + dists[longest_idx+1:]), new_dist)
                        if new_max < best_max:
                            best_max = new_max
                            routes[longest_idx] = new_route
                            dists[longest_idx] = new_dist
                            improved = True
                            report_best_vrp([list(r) for r in routes])
                            break
                if improved:
                    break
        if not improved:
            # Ruin and recreate: remove customer with largest insertion cost from longest route
            longest_route = routes[longest_idx]
            if len(longest_route) <= 3:
                break
            # Compute insertion cost contribution for each internal customer
            contributions = []
            for idx in range(1, len(longest_route)-1):
                prev = longest_route[idx-1]
                cust = longest_route[idx]
                next_cust = longest_route[idx+1]
                contrib = distance_matrix[prev, cust] + distance_matrix[cust, next_cust] - distance_matrix[prev, next_cust]
                contributions.append((contrib, cust, idx))
            if not contributions:
                break
            # Pick customer with largest contribution (most costly)
            _, worst_cust, worst_pos = max(contributions, key=lambda x: x[0])
            # Remove from its route
            new_route = [x for x in longest_route if x != worst_cust]
            old_dist = dists[longest_idx]
            new_dist = route_dist(new_route)
            # Reinsert using regret into any route (including possibly back to same route at different position)
            # Use regret insertion across all routes
            costs = []
            for r_idx, route in enumerate(routes):
                cost, pos = best_insertion(worst_cust, route)
                costs.append((cost, r_idx, pos))
            costs.sort(key=lambda x: x[0])
            if len(costs) == 1:
                best_regret = costs[0][0] * 2
            else:
                best_regret = costs[1][0] - costs[0][0]
            best_choice = costs[0]  # best cost
            # Apply the move even if it doesn't immediately improve max? But we want to escape, so we do it unconditionally but then update best_max
            # Actually, we should only apply if it improves? Or we accept a temporary worsening to escape? We'll apply and recompute max; if worse, we revert? 
            # Let's apply unconditionally to explore, but then we might degrade. To keep monotonic improvement, we'll only apply if it does not increase best_max? Or we can accept if new_max <= best_max * 1.05? But the instruction says exploitation, so we should be conservative.
            # We'll only apply if the new max is <= best_max, else revert.
            new_routes = [list(r) for r in routes]
            new_routes[longest_idx] = new_route
            new_routes[best_choice[1]] = list(new_routes[best_choice[1]])
            new_routes[best_choice[1]].insert(best_choice[2], worst_cust)
            new_dists = [route_dist(r) for r in new_routes]
            new_max = max(new_dists)
            if new_max < best_max:
                routes = new_routes
                dists = new_dists
                best_max = new_max
                improved = True
                report_best_vrp([list(r) for r in routes])
            # else do nothing
        if not improved:
            break
    # Ensure exactly truck_count routes
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

def report_best_vrp(routes):
    # Placeholder: could log or store best solution
    pass