import numpy as np
import math
import random

def report_best_vrp(routes):
    """Placeholder; actual function is provided by the environment."""
    pass

def route_distance(route, dm):
    if len(route) < 2:
        return 0.0
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def total_distance(routes, dm):
    return sum(route_distance(r, dm) for r in routes)

def max_route_distance(routes, dm):
    return max(route_distance(r, dm) for r in routes)

def clarke_wright_init(dm, truck_count):
    n = dm.shape[0]
    customers = list(range(1, n))
    # Each customer starts as a route [0, customer, 0]
    routes = [[0, c, 0] for c in customers]
    while len(routes) > truck_count:
        # Compute savings for merging
        best_sav = -float('inf')
        best_pair = None
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                # Find endpoints that connect to depot
                # We can only merge if both routes have at least one interior node
                if len(routes[i]) <= 2 or len(routes[j]) <= 2:
                    continue
                # Endpoints: first interior and last interior (after depot at start/end)
                # Actually we need the last customer before depot in route i and first customer after depot in route j
                # For route i, the last customer is routes[i][-2]
                # For route j, the first customer is routes[j][1]
                # But also we could connect from end of i to start of j or start of i to end of j
                # Typical savings: s(i_last, j_first) = dm[0][i_last] + dm[0][j_first] - dm[i_last][j_first]
                i_last = routes[i][-2]
                j_first = routes[j][1]
                sav = dm[0][i_last] + dm[0][j_first] - dm[i_last][j_first]
                if sav > best_sav:
                    best_sav = sav
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        # Merge routes[i] and routes[j] by connecting i's last to j's first
        i_last = routes[i][-2]
        j_first = routes[j][1]
        if dm[0][i_last] + dm[0][j_first] - dm[i_last][j_first] <= 0:
            # No benefit, merge anyway to reduce routes
            pass
        new_route = routes[i][:-1] + routes[j][1:]
        # Remove old routes and add new
        routes.pop(j)
        routes.pop(i)
        routes.append(new_route)
        if len(routes) <= truck_count:
            break
    # If we have fewer than truck_count, add empty routes
    while len(routes) < truck_count:
        routes.append([0, 0])
    return routes

def intra_2opt(route, dm):
    best_route = route[:]
    best_dist = route_distance(route, dm)
    improved = False
    for i in range(1, len(route)-2):
        for j in range(i+1, len(route)-1):
            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
            new_dist = route_distance(new_route, dm)
            if new_dist < best_dist - 1e-12:
                best_dist = new_dist
                best_route = new_route
                improved = True
    return best_route, improved

def relocate_between_routes(routes, from_idx, to_idx, cust_pos, dm):
    # from_idx: index of source route, cust_pos: position of customer to move (1-indexed within route)
    # to_idx: destination route index
    # Returns new routes list if feasible, else None
    if from_idx == to_idx:
        return None
    route_from = routes[from_idx]
    if len(route_from) <= 3:  # depot-customer-depot
        return None
    cust = route_from[cust_pos]
    if cust == 0:
        return None
    # Remove customer from from_route
    new_from = route_from[:cust_pos] + route_from[cust_pos+1:]
    # Insert into to_route in best position to minimize its distance
    route_to = routes[to_idx]
    best_dist = float('inf')
    best_pos = 1
    for pos in range(1, len(route_to)):
        new_to = route_to[:pos] + [cust] + route_to[pos:]
        d = route_distance(new_to, dm)
        if d < best_dist:
            best_dist = d
            best_pos = pos
    new_to = route_to[:best_pos] + [cust] + route_to[best_pos:]
    # Build new routes list
    new_routes = [r[:] for r in routes]
    new_routes[from_idx] = new_from
    new_routes[to_idx] = new_to
    return new_routes

def swap_between_routes(routes, idx1, pos1, idx2, pos2, dm):
    if idx1 == idx2:
        return None
    if len(routes[idx1]) <= 3 or len(routes[idx2]) <= 3:
        return None
    cust1 = routes[idx1][pos1]
    cust2 = routes[idx2][pos2]
    if cust1 == 0 or cust2 == 0:
        return None
    # Remove both customers
    new1 = routes[idx1][:pos1] + routes[idx1][pos1+1:]
    new2 = routes[idx2][:pos2] + routes[idx2][pos2+1:]
    # Insert cust2 into new1 at best position
    best_dist1 = float('inf')
    best_pos1 = 1
    for pos in range(1, len(new1)):
        temp = new1[:pos] + [cust2] + new1[pos:]
        d = route_distance(temp, dm)
        if d < best_dist1:
            best_dist1 = d
            best_pos1 = pos
    new1 = new1[:best_pos1] + [cust2] + new1[best_pos1:]
    # Insert cust1 into new2 at best position
    best_dist2 = float('inf')
    best_pos2 = 1
    for pos in range(1, len(new2)):
        temp = new2[:pos] + [cust1] + new2[pos:]
        d = route_distance(temp, dm)
        if d < best_dist2:
            best_dist2 = d
            best_pos2 = pos
    new2 = new2[:best_pos2] + [cust1] + new2[best_pos2:]
    new_routes = [r[:] for r in routes]
    new_routes[idx1] = new1
    new_routes[idx2] = new2
    return new_routes

def cross_2opt_star(routes, idx1, idx2, dm):
    # Exchange segments between two routes
    if idx1 == idx2:
        return None
    r1 = routes[idx1]
    r2 = routes[idx2]
    if len(r1) <= 2 or len(r2) <= 2:
        return None
    best_routes = None
    best_max = float('inf')
    # For each pair of breakpoints (i in r1, j in r2) where i,j are positions after a customer (including after last)
    # Actually we can consider i in 1..len(r1)-2, j in 1..len(r2)-2
    for i in range(1, len(r1)-1):
        for j in range(1, len(r2)-1):
            # new r1: r1[0..i] + r2[j+1..]   (depot to depot)
            # new r2: r2[0..j] + r1[i+1..]
            new1 = r1[:i+1] + r2[j+1:]
            new2 = r2[:j+1] + r1[i+1:]
            # Ensure they start and end with depot
            if new1[0] != 0:
                new1 = [0] + new1
            if new1[-1] != 0:
                new1.append(0)
            if new2[0] != 0:
                new2 = [0] + new2
            if new2[-1] != 0:
                new2.append(0)
            # Check for duplicate customers
            if len(set(new1[1:-1] + new2[1:-1])) != len(new1[1:-1]) + len(new2[1:-1]):
                continue
            new_routes = [r[:] for r in routes]
            new_routes[idx1] = new1
            new_routes[idx2] = new2
            new_max = max_route_distance(new_routes, dm)
            if new_max < best_max - 1e-12:
                best_max = new_max
                best_routes = new_routes
    return best_routes

def perturbation(routes, dm, num_eject=2):
    # Find longest route
    max_dist = max_route_distance(routes, dm)
    long_idx = None
    for idx, r in enumerate(routes):
        if route_distance(r, dm) == max_dist:
            long_idx = idx
            break
    if long_idx is None or len(routes[long_idx]) <= 3:
        return routes
    long_route = routes[long_idx]
    # Compute delta for each customer: cost if removed
    deltas = []
    for pos in range(1, len(long_route)-1):
        cust = long_route[pos]
        prev = long_route[pos-1]
        nxt = long_route[pos+1]
        delta = dm[prev][cust] + dm[cust][nxt] - dm[prev][nxt]
        deltas.append((delta, pos, cust))
    # Eject customers with smallest delta (least removal cost)
    deltas.sort(key=lambda x: x[0])
    ejected = []
    new_route = long_route[:]
    for _, pos, cust in deltas[:num_eject]:
        # Remove from new_route
        idx = new_route.index(cust)
        del new_route[idx]
        ejected.append(cust)
    # Reinsert ejected customers greedily minimizing max distance
    new_routes = [r[:] for r in routes]
    new_routes[long_idx] = new_route
    for cust in ejected:
        best_max = float('inf')
        best_new_routes = None
        for t in range(len(new_routes)):
            route = new_routes[t]
            for pos in range(1, len(route)):
                temp_route = route[:pos] + [cust] + route[pos:]
                temp_routes = new_routes[:]
                temp_routes[t] = temp_route
                new_max = max_route_distance(temp_routes, dm)
                if new_max < best_max - 1e-12:
                    best_max = new_max
                    best_new_routes = temp_routes[:]
        if best_new_routes is not None:
            new_routes = best_new_routes
    return new_routes

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    np.random.seed(0)
    dm = distance_matrix
    n = dm.shape[0]
    # Initialisation
    routes = clarke_wright_init(dm, truck_count)
    best_routes = [r[:] for r in routes]
    best_max = max_route_distance(routes, dm)
    report_best_vrp(best_routes)
    
    # Parameters
    max_iter = 1000
    max_no_improve = 200
    temp_init = 1000.0
    temp_final = 1.0
    cooling = 0.99
    temp = temp_init
    no_improve_count = 0
    
    for iteration in range(max_iter):
        # Identify longest route
        current_max = max_route_distance(routes, dm)
        long_candidates = [i for i, r in enumerate(routes) if route_distance(r, dm) == current_max]
        if not long_candidates:
            continue
        long_idx = random.choice(long_candidates)
        
        # Try improvements
        improved = False
        # Intra 2-opt on longest
        new_route, imp = intra_2opt(routes[long_idx], dm)
        if imp:
            new_routes = [r[:] for r in routes]
            new_routes[long_idx] = new_route
            new_max = max_route_distance(new_routes, dm)
            if new_max < current_max - 1e-12:  # strict improvement
                routes = new_routes
                improved = True
        
        if not improved:
            # Inter relocate (move from longest to another)
            best_new_max = float('inf')
            best_new_routes = None
            for pos in range(1, len(routes[long_idx])-1):
                cust = routes[long_idx][pos]
                if cust == 0:
                    continue
                for t in range(truck_count):
                    if t == long_idx:
                        continue
                    new_routes = relocate_between_routes(routes, long_idx, t, pos, dm)
                    if new_routes is None:
                        continue
                    new_max = max_route_distance(new_routes, dm)
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_new_routes = new_routes
            if best_new_routes is not None and best_new_max < current_max:
                routes = best_new_routes
                improved = True
        
        if not improved:
            # Inter swap
            best_new_max = float('inf')
            best_new_routes = None
            for pos1 in range(1, len(routes[long_idx])-1):
                cust1 = routes[long_idx][pos1]
                if cust1 == 0:
                    continue
                for t in range(truck_count):
                    if t == long_idx:
                        continue
                    for pos2 in range(1, len(routes[t])-1):
                        cust2 = routes[t][pos2]
                        if cust2 == 0:
                            continue
                        new_routes = swap_between_routes(routes, long_idx, pos1, t, pos2, dm)
                        if new_routes is None:
                            continue
                        new_max = max_route_distance(new_routes, dm)
                        if new_max < best_new_max - 1e-12:
                            best_new_max = new_max
                            best_new_routes = new_routes
            if best_new_routes is not None and best_new_max < current_max:
                routes = best_new_routes
                improved = True
        
        if not improved:
            # Cross 2-opt*
            best_new_max = float('inf')
            best_new_routes = None
            for t in range(truck_count):
                if t == long_idx:
                    continue
                new_routes = cross_2opt_star(routes, long_idx, t, dm)
                if new_routes is None:
                    continue
                new_max = max_route_distance(new_routes, dm)
                if new_max < best_new_max - 1e-12:
                    best_new_max = new_max
                    best_new_routes = new_routes
            if best_new_routes is not None and best_new_max < current_max:
                routes = best_new_routes
                improved = True
        
        # Simulated annealing acceptance of worse moves
        if not improved:
            # Perturb if stuck
            if no_improve_count > 20:
                routes = perturbation(routes, dm, num_eject=min(3, len(routes[long_idx])-2))
                current_max = max_route_distance(routes, dm)
                no_improve_count = 0
            else:
                # Random small perturbation: relocate a random customer
                # Choose a random route and a random customer, move to a random position
                from_idx = random.randrange(truck_count)
                if len(routes[from_idx]) > 3:
                    pos = random.randint(1, len(routes[from_idx])-2)
                    to_idx = random.randrange(truck_count)
                    if to_idx == from_idx:
                        to_idx = (to_idx + 1) % truck_count
                    new_routes = relocate_between_routes(routes, from_idx, to_idx, pos, dm)
                    if new_routes is not None:
                        new_max = max_route_distance(new_routes, dm)
                        if new_max < current_max or random.random() < math.exp((current_max - new_max)/temp):
                            routes = new_routes
                            current_max = new_max
        
        # Update best
        new_max = max_route_distance(routes, dm)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Cooling
        temp = temp * cooling
        if temp < temp_final:
            temp = temp_final
        
        # Early termination
        if no_improve_count >= max_no_improve:
            break
    
    # Ensure exactly truck_count routes and all customers present
    used_customers = set()
    for r in best_routes:
        used_customers.update(r[1:-1])
    missing = set(range(1, n)) - used_customers
    if missing:
        # Should not happen, but just in case, assign missing customers arbitrarily
        for c in missing:
            # Insert into shortest route
            min_len = float('inf')
            min_idx = 0
            for idx, r in enumerate(best_routes):
                d = route_distance(r, dm)
                if d < min_len:
                    min_len = d
                    min_idx = idx
            # Insert at best position
            route = best_routes[min_idx]
            best_pos = 1
            best_dist = float('inf')
            for pos in range(1, len(route)):
                temp = route[:pos] + [c] + route[pos:]
                d = route_distance(temp, dm)
                if d < best_dist:
                    best_dist = d
                    best_pos = pos
            best_routes[min_idx] = route[:best_pos] + [c] + route[best_pos:]
    # Ensure each route starts and ends with 0
    for i in range(len(best_routes)):
        if best_routes[i][0] != 0:
            best_routes[i].insert(0, 0)
        if best_routes[i][-1] != 0:
            best_routes[i].append(0)
    # Pad with empty routes
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    # Remove duplicate 0s if any
    for i in range(len(best_routes)):
        if len(best_routes[i]) > 2:
            if best_routes[i][0] == 0 and best_routes[i][1] == 0:
                best_routes[i] = [0] + best_routes[i][2:]
    return best_routes
}