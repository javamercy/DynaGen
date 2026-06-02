import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    
    customers = list(range(1, n))
    # Step 1: Clarke-Wright savings
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
            savings.append((s, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))
    
    # Initialize each customer as its own route
    routes = [[0, c, 0] for c in customers]
    # map customer to route index
    cust_to_route = {c: idx for idx, c in enumerate(customers)}
    
    for s, i, j in savings:
        if cust_to_route[i] == cust_to_route[j]:
            continue
        ri = cust_to_route[i]
        rj = cust_to_route[j]
        # Check if i and j are endpoints of their routes (adjacent to depot)
        route_i = routes[ri]
        route_j = routes[rj]
        # Endpoints: second element and second-last element (since routes start and end with 0)
        endpoints_i = [route_i[1], route_i[-2]]
        endpoints_j = [route_j[1], route_j[-2]]
        if i not in endpoints_i or j not in endpoints_j:
            continue
        # Determine orientation: we can merge if one endpoint is at start and other at end
        # Actually, we can connect i's route end and j's route end or start
        # For simplicity, we'll connect if i is at the beginning (route_i[1]==i) and j at the end (route_j[-2]==j)
        # or vice versa. But we can handle all four cases.
        # Case 1: i at start of route_i, j at end of route_j -> append route_i reversed? Actually we want to merge without reversal.
        # Since we don't care about direction, we can always ensure i is at start and j at end by reversing if needed.
        # For simplicity, we'll reverse route_i if i is at the end, so that i becomes start.
        if route_i[-2] == i:
            # reverse route_i (excluding 0s) but keep 0s at ends
            route_i[1:-1] = route_i[-2:0:-1]
            # now i should be at route_i[1]
        if route_j[1] == j:
            # reverse route_j so that j is at end
            route_j[1:-1] = route_j[-2:0:-1]
        # Now i is at start of route_i, j at end of route_j
        if route_i[1] == i and route_j[-2] == j:
            # Merge: concatenate route_i (without closing 0) and route_j (without opening 0)
            new_route = route_i[:-1] + route_j[1:]
            # Remove old routes
            # We'll replace route_i with new route, and mark route_j as empty
            routes[ri] = new_route
            routes[rj] = [0, 0]
            # Update cust_to_route: all customers from rj now in ri
            for c in route_j[1:-1]:
                cust_to_route[c] = ri
    
    # Remove empty routes and pad to truck_count
    non_empty = [r for r in routes if len(r) > 2]
    empty_count = truck_count - len(non_empty)
    if empty_count < 0:
        # Merge excess routes into existing ones? But we should have at most truck_count routes if we merged correctly.
        # If more routes than trucks, combine smallest routes?
        # For safety, we'll just take first truck_count routes, but this may cause infeasibility. Better to ensure we have exactly truck_count.
        # We'll sort by number of customers descending and keep truck_count, merging the rest into the largest.
        non_empty.sort(key=lambda r: -len(r))
        routes = non_empty[:truck_count]
        extra = non_empty[truck_count:]
        for ext in extra:
            # merge ext into the shortest route
            shortest_idx = min(range(truck_count), key=lambda i: len(routes[i]))
            # simply append customers
            for c in ext[1:-1]:
                routes[shortest_idx].insert(-1, c)
    else:
        routes = non_empty + [[0, 0] for _ in range(empty_count)]
    
    # Ensure exactly truck_count routes
    while len(routes) < truck_count:
        routes.append([0, 0])
    # Trim if too many
    routes = routes[:truck_count]
    
    # Helper to compute distance of a route
    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    # Step 2: Intra-route 2-opt on each route
    for idx, route in enumerate(routes):
        if len(route) <= 3:
            continue
        improved = True
        max_iters = len(route) * len(route)
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            best_gain = 0
            best_ij = None
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # gain = old - new
                    old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                    new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    gain = old - new
                    if gain > best_gain:
                        best_gain = gain
                        best_ij = (i, j)
            if best_gain > 0:
                i, j = best_ij
                route[i:j+1] = route[i:j+1][::-1]
                improved = True
    
    # Step 3: Inter-route improvement focusing on max route distance
    def total_customers():
        cnt = 0
        for r in routes:
            cnt += len(r) - 2
        return cnt
    
    max_iter = 2 * total_customers()
    for iteration in range(max_iter):
        dists = [route_distance(r) for r in routes]
        max_dist = max(dists)
        max_idx = dists.index(max_dist)
        best_improvement = None
        best_new_max = max_dist
        # Try moving a customer from max route to another route
        max_route = routes[max_idx]
        # List customers (excluding depot)
        cust_list = max_route[1:-1]
        for c in cust_list:
            for other_idx in range(len(routes)):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                # Try inserting c in each position in other route (between 0 and 0)
                for pos in range(1, len(other_route)):
                    # Simulate new routes
                    new_max_route = [x for x in max_route if x != c]
                    new_other_route = other_route[:pos] + [c] + other_route[pos:]
                    # Ensure length >2 (if other route was empty, it becomes [0, c, 0])
                    if len(new_other_route) < 3:
                        new_other_route = [0, c, 0]  # but pos should be 1, so it's okay
                    new_max_dist = route_distance(new_max_route)
                    new_other_dist = route_distance(new_other_route)
                    candidate_max = max(new_max_dist, new_other_dist)
                    if candidate_max < best_new_max:
                        best_new_max = candidate_max
                        best_improvement = ('move', c, other_idx, pos, new_max_route, new_other_route)
        # Also try swapping customers between max route and other routes
        if len(cust_list) > 0:
            for c in cust_list:
                for other_idx in range(len(routes)):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    other_custs = other_route[1:-1]
                    for d in other_custs:
                        # Swap c and d
                        new_max_route = [d if x==c else x for x in max_route]
                        new_other_route = [c if x==d else x for x in other_route]
                        new_max_dist = route_distance(new_max_route)
                        new_other_dist = route_distance(new_other_route)
                        candidate_max = max(new_max_dist, new_other_dist)
                        if candidate_max < best_new_max:
                            best_new_max = candidate_max
                            best_improvement = ('swap', c, d, other_idx, new_max_route, new_other_route)
        if best_improvement is None:
            break
        # Apply improvement
        op = best_improvement[0]
        if op == 'move':
            _, c, other_idx, pos, new_max, new_other = best_improvement
            routes[max_idx] = new_max
            routes[other_idx] = new_other
        else:  # swap
            _, c, d, other_idx, new_max, new_other = best_improvement
            routes[max_idx] = new_max
            routes[other_idx] = new_other
        # Report if better
        dists = [route_distance(r) for r in routes]
        new_max_dist = max(dists)
        if new_max_dist < max_dist:
            report_best_vrp(routes)
    
    # Final check: ensure each route starts and ends with 0
    for r in routes:
        if r[0] != 0:
            r.insert(0, 0)
        if r[-1] != 0:
            r.append(0)
    # Ensure exactly truck_count routes
    if len(routes) < truck_count:
        routes.extend([[0, 0]] * (truck_count - len(routes)))
    routes = routes[:truck_count]
    return routes