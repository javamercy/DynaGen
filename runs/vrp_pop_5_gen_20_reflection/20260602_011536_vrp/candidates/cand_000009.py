import numpy as np

n = distance_matrix.shape[0]
dist = distance_matrix.tolist()
n_customers = n - 1

# Construction: cheapest insertion minimizing max route distance
routes = [[0, 0] for _ in range(truck_count)]
unassigned = list(range(1, n))

while unassigned:
    best_max = float('inf')
    best_total = float('inf')
    best_node = None
    best_route = None
    best_pos = None
    for node in unassigned:
        for r in range(truck_count):
            route = routes[r]
            for pos in range(1, len(route)):
                # compute new route distance after insertion
                new_route_dist = 0
                prev = route[0]
                for k in range(1, len(route)):
                    if k == pos:
                        new_route_dist += dist[prev][node]
                        prev = node
                    new_route_dist += dist[prev][route[k]]
                    prev = route[k]
                # compute new max across routes
                current_max = 0
                for rr in range(truck_count):
                    if rr == r:
                        route_dist = new_route_dist
                    else:
                        route_dist = sum(dist[routes[rr][i]][routes[rr][i+1]] for i in range(len(routes[rr])-1))
                    if route_dist > current_max:
                        current_max = route_dist
                # tie-breaking: smaller max, then smaller total distance of the modified route
                if current_max < best_max or (current_max == best_max and new_route_dist < best_total):
                    best_max = current_max
                    best_total = new_route_dist
                    best_node = node
                    best_route = r
                    best_pos = pos
    # insert best node
    routes[best_route].insert(best_pos, best_node)
    unassigned.remove(best_node)

def route_distance(route):
    return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))

def max_route_distance(routes):
    return max(route_distance(r) for r in routes)

best_obj = max_route_distance(routes)
report_best_vrp([list(r) for r in routes])

# Local search with adaptive focus on longest route
max_passes = 20
max_restarts = 5
restart_count = 0

while restart_count <= max_restarts:
    improved = False
    for _ in range(max_passes):
        # Identify the route with maximum distance
        max_dist = 0
        max_route_idx = None
        for r, route in enumerate(routes):
            d = route_distance(route)
            if d > max_dist:
                max_dist = d
                max_route_idx = r
        
        # First-improvement: try moves from the longest route
        # Relocate: move a customer from longest route to another route
        if max_route_idx is not None:
            route_long = routes[max_route_idx]
            # iterate over nodes in longest route (excluding depots)
            for idx in range(1, len(route_long)-1):
                node = route_long[idx]
                # try moving node to other routes
                for r in range(truck_count):
                    if r == max_route_idx:
                        continue
                    route_other = routes[r]
                    for pos in range(1, len(route_other)):
                        # build new routes
                        new_route_long = route_long[:idx] + route_long[idx+1:]
                        if len(new_route_long) < 2:
                            new_route_long = [0, 0]
                        new_route_other = route_other[:pos] + [node] + route_other[pos:]
                        new_routes = [list(routes[i]) for i in range(truck_count)]
                        new_routes[max_route_idx] = new_route_long
                        new_routes[r] = new_route_other
                        new_obj = max_route_distance(new_routes)
                        if new_obj < best_obj:
                            routes = new_routes
                            best_obj = new_obj
                            improved = True
                            report_best_vrp([list(r) for r in routes])
                            break
                    if improved:
                        break
                if improved:
                    break
        if improved:
            continue
        
        # Swap: try swapping a customer from longest route with a customer from another route
        if max_route_idx is not None:
            route_long = routes[max_route_idx]
            for idx_i in range(1, len(route_long)-1):
                node_i = route_long[idx_i]
                for r in range(truck_count):
                    if r == max_route_idx:
                        continue
                    route_other = routes[r]
                    for idx_j in range(1, len(route_other)-1):
                        node_j = route_other[idx_j]
                        # build new routes after swap
                        # remove i from long, j from other
                        new_route_long = route_long[:idx_i] + route_long[idx_i+1:]
                        if len(new_route_long) < 2:
                            new_route_long = [0, 0]
                        new_route_other = route_other[:idx_j] + route_other[idx_j+1:]
                        if len(new_route_other) < 2:
                            new_route_other = [0, 0]
                        # insert i into other, j into long
                        # try all positions (for simplicity, we insert at the end of the shortened route? To keep bounded, we try positions 1..len-1.
                        # But to reduce complexity, we consider inserting at the end of each route (position = len). However, best improvement may require different positions.
                        # We'll do a nested loop over insertion positions? That would be heavy. Instead, we just insert at the best position according to local gain?
                        # For simplicity, we consider insertion at the position that minimizes the new route distance (greedy).
                        best_new_long = new_route_long[:1] + [node_j] + new_route_long[1:]
                        best_new_other = new_route_other[:1] + [node_i] + new_route_other[1:]
                        # try all insertion positions for j in long:
                        best_long_dist = float('inf')
                        best_new_long = None
                        for pos_i in range(1, len(new_route_long)):
                            trial = new_route_long[:pos_i] + [node_j] + new_route_long[pos_i:]
                            d = route_distance(trial)
                            if d < best_long_dist:
                                best_long_dist = d
                                best_new_long = trial
                        best_other_dist = float('inf')
                        best_new_other = None
                        for pos_j in range(1, len(new_route_other)):
                            trial = new_route_other[:pos_j] + [node_i] + new_route_other[pos_j:]
                            d = route_distance(trial)
                            if d < best_other_dist:
                                best_other_dist = d
                                best_new_other = trial
                        new_routes = [list(routes[i]) for i in range(truck_count)]
                        new_routes[max_route_idx] = best_new_long
                        new_routes[r] = best_new_other
                        new_obj = max_route_distance(new_routes)
                        if new_obj < best_obj:
                            routes = new_routes
                            best_obj = new_obj
                            improved = True
                            report_best_vrp([list(r) for r in routes])
                            break
                if improved:
                    break
        if improved:
            continue
        
        # 2-opt within each route
        for r in range(truck_count):
            route = routes[r]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    other_max = 0
                    for rr in range(truck_count):
                        if rr != r:
                            d = route_distance(routes[rr])
                            if d > other_max:
                                other_max = d
                    new_obj = max(new_dist, other_max)
                    if new_obj < best_obj:
                        routes[r] = new_route
                        best_obj = new_obj
                        improved = True
                        report_best_vrp([list(r) for r in routes])
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        
        # Cross-route 2-opt (2-opt*)
        for r1 in range(truck_count):
            route1 = routes[r1]
            if len(route1) <= 3:
                continue
            for r2 in range(r1+1, truck_count):
                route2 = routes[r2]
                if len(route2) <= 3:
                    continue
                for a in range(1, len(route1)-2):
                    for b in range(1, len(route2)-2):
                        new_route1 = route1[:a+1] + route2[b+1:]
                        new_route2 = route2[:b+1] + route1[a+1:]
                        new_dist1 = route_distance(new_route1)
                        new_dist2 = route_distance(new_route2)
                        other_max = 0
                        for rr in range(truck_count):
                            if rr != r1 and rr != r2:
                                d = route_distance(routes[rr])
                                if d > other_max:
                                    other_max = d
                        new_obj = max(new_dist1, new_dist2, other_max)
                        if new_obj < best_obj:
                            routes[r1] = new_route1
                            routes[r2] = new_route2
                            best_obj = new_obj
                            improved = True
                            report_best_vrp([list(r) for r in routes])
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break  # no improvement in this pass
    
    if not improved:
        # restart: random move from longest route
        # identify longest route again
        max_dist = 0
        max_route_idx = None
        for r, route in enumerate(routes):
            d = route_distance(route)
            if d > max_dist:
                max_dist = d
                max_route_idx = r
        if max_route_idx is not None and len(routes[max_route_idx]) > 2:
            # pick a random customer from longest route (excluding depots)
            import random
            idx = random.randrange(1, len(routes[max_route_idx])-1)
            node = routes[max_route_idx][idx]
            # remove node from its route
            new_route_long = routes[max_route_idx][:idx] + routes[max_route_idx][idx+1:]
            if len(new_route_long) < 2:
                new_route_long = [0, 0]
            # insert into a random other route at best position
            # choose a random different route
            other_r = random.choice([r for r in range(truck_count) if r != max_route_idx])
            route_other = routes[other_r]
            best_pos = None
            best_increase = float('inf')
            for pos in range(1, len(route_other)):
                trial_other = route_other[:pos] + [node] + route_other[pos:]
                d = route_distance(trial_other)
                if d < best_increase:
                    best_increase = d
                    best_pos = pos
            new_route_other = route_other[:best_pos] + [node] + route_other[best_pos:]
            routes[max_route_idx] = new_route_long
            routes[other_r] = new_route_other
            best_obj = max_route_distance(routes)
            report_best_vrp([list(r) for r in routes])
            restart_count += 1
            # continue outer while
        else:
            break
    else:
        # improvement found, reset restart count? We'll keep it simple: only restart when no improvement in a full set of passes
        pass

return [list(r) for r in routes]