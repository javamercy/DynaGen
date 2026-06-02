import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    # initialize routes as [0,0] for each truck
    routes = [[0, 0] for _ in range(truck_count)]
    route_distances = [0.0 for _ in range(truck_count)]
    
    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    for r in range(truck_count):
        route_distances[r] = compute_route_distance(routes[r])
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_distances)
    
    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        current_max = max(route_distances)
        if current_max < best_max:
            best_max = current_max
            best_routes = [list(r) for r in routes]
    
    # Regret-2 insertion
    def regret2_insertion(routes, route_distances, unassigned):
        while unassigned:
            best_customer = None
            best_regret = -1.0
            best_insertion = None
            for c in unassigned:
                costs = []
                for r_idx, route in enumerate(routes):
                    curr_dist = route_distances[r_idx]
                    for i in range(1, len(route)):
                        new_dist = curr_dist - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], c] + distance_matrix[c, route[i]]
                        other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0.0)
                        cand_max = max(new_dist, other_max)
                        costs.append((cand_max, r_idx, i))
                costs.sort(key=lambda x: (x[0], x[1], x[2]))
                if len(costs) >= 2:
                    regret = costs[1][0] - costs[0][0]
                else:
                    regret = 0.0
                if regret > best_regret or (regret == best_regret and (best_customer is None or c < best_customer)):
                    best_regret = regret
                    best_customer = c
                    best_insertion = costs[0][1:]
            r_idx, i = best_insertion
            route = routes[r_idx]
            route.insert(i, best_customer)
            route_distances[r_idx] = compute_route_distance(route)
            unassigned.remove(best_customer)
    
    unassigned = set(range(1, n))
    regret2_insertion(routes, route_distances, unassigned)
    report_best_vrp(routes)
    
    # Multi-neighborhood local search (2-opt, swap, relocate) until no improvement
    def local_search(routes, route_distances, max_iterations=100):
        for _ in range(max_iterations):
            improved = False
            # Intra-route 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_route_distance(new_route)
                        other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0.0)
                        new_max = max(new_dist, other_max)
                        if new_max < best_max:
                            routes[r_idx] = new_route
                            route_distances[r_idx] = new_dist
                            report_best_vrp(routes)
                            improved = True
            # Inter-route swap
            for r1 in range(truck_count):
                for r2 in range(r1+1, truck_count):
                    route1 = routes[r1]
                    route2 = routes[r2]
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i] + [route2[j]] + route1[i+1:]
                            new2 = route2[:j] + [route1[i]] + route2[j+1:]
                            new_dist1 = compute_route_distance(new1)
                            new_dist2 = compute_route_distance(new2)
                            other_max = max(route_distances[:r1] + route_distances[r1+1:r2] + route_distances[r2+1:], default=0.0)
                            new_max = max(new_dist1, new_dist2, other_max)
                            if new_max < best_max:
                                routes[r1] = new1
                                routes[r2] = new2
                                route_distances[r1] = new_dist1
                                route_distances[r2] = new_dist2
                                report_best_vrp(routes)
                                improved = True
            # Intra-route relocate (move a customer within its route)
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for i in range(1, len(route)-1):
                    customer = route[i]
                    for j in range(1, len(route)-1):
                        if i == j or i+1 == j:
                            continue
                        new_route = route[:i] + route[i+1:]
                        new_route.insert(j, customer)
                        if compute_route_distance(new_route) < route_distances[r_idx]:
                            # compute new max
                            new_dist = compute_route_distance(new_route)
                            other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0.0)
                            new_max = max(new_dist, other_max)
                            if new_max < best_max:
                                routes[r_idx] = new_route
                                route_distances[r_idx] = new_dist
                                report_best_vrp(routes)
                                improved = True
                                break
                    if improved:
                        break
            # Inter-route relocate (move customer from one route to another)
            for r1 in range(truck_count):
                for r2 in range(truck_count):
                    if r1 == r2:
                        continue
                    route1 = routes[r1]
                    route2 = routes[r2]
                    for i in range(1, len(route1)-1):
                        customer = route1[i]
                        for j in range(1, len(route2)):
                            new1 = route1[:i] + route1[i+1:]
                            new2 = route2[:j] + [customer] + route2[j:]
                            new_dist1 = compute_route_distance(new1)
                            new_dist2 = compute_route_distance(new2)
                            other_max = max(route_distances[:r1] + route_distances[r1+1:r2] + route_distances[r2+1:], default=0.0)
                            new_max = max(new_dist1, new_dist2, other_max)
                            if new_max < best_max:
                                routes[r1] = new1
                                routes[r2] = new2
                                route_distances[r1] = new_dist1
                                route_distances[r2] = new_dist2
                                report_best_vrp(routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break
        return routes, route_distances
    
    routes, route_distances = local_search(routes, route_distances)
    
    # Ruin-recreate with multiple restarts
    for restart in range(5):
        # random seed for each restart to ensure diversity
        random.seed(restart)
        # backup current best routes
        current_routes = [list(r) for r in best_routes]
        current_distances = [compute_route_distance(r) for r in current_routes]
        
        # Ruin: remove approximately 30% of customers, focusing on longest routes
        num_to_remove = max(1, int(0.3 * (n-1)))
        # sorted routes by distance descending
        sorted_route_indices = sorted(range(truck_count), key=lambda x: current_distances[x], reverse=True)
        customers_to_remove = []
        # first take from longest routes until we have enough
        for r_idx in sorted_route_indices:
            route = current_routes[r_idx]
            # remove all interior customers (positions 1 to -2)
            removable = route[1:-1]
            random.shuffle(removable)
            for c in removable:
                if len(customers_to_remove) >= num_to_remove:
                    break
                customers_to_remove.append(c)
            if len(customers_to_remove) >= num_to_remove:
                break
        # If still not enough, take from remaining routes
        if len(customers_to_remove) < num_to_remove:
            for r_idx in sorted_route_indices:
                route = current_routes[r_idx]
                removable = route[1:-1]
                random.shuffle(removable)
                for c in removable:
                    if c not in customers_to_remove and len(customers_to_remove) < num_to_remove:
                        customers_to_remove.append(c)
        # Actually remove them
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            current_routes[r_idx] = [0] + [c for c in route[1:-1] if c not in customers_to_remove] + [0]
            current_distances[r_idx] = compute_route_distance(current_routes[r_idx])
        unassigned = set(customers_to_remove)
        # Reinsert using regret-2 on the current state
        regret2_insertion(current_routes, current_distances, unassigned)
        # Apply local search to this new solution
        current_routes, current_distances = local_search(current_routes, current_distances)
        # update best if improved
        current_max = max(current_distances)
        if current_max < best_max:
            best_max = current_max
            best_routes = [list(r) for r in current_routes]
    
    return best_routes