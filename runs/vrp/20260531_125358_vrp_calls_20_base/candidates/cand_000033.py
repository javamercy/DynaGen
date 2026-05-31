import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n:
        routes = [[0, i, 0] for i in customers]
        routes += [[0,0] for _ in range(truck_count - len(customers))]
        return routes

    def route_dist(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    best_routes = None
    best_max = float('inf')
    num_restarts = 10
    for _ in range(num_restarts):
        # Initialize each customer as a separate route
        routes = [[0, i, 0] for i in customers]
        # Compute savings
        savings = []
        for i in range(1, n):
            for j in range(i+1, n):
                s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                savings.append((s, i, j))
        savings.sort(reverse=True, key=lambda x: (x[0], -x[1], -x[2]))  # deterministic tie: larger i,j first? but we randomize later
        # Keep track of which route each customer is in, and the free endpoints (adjacent to depot)
        cust_to_route = {i: i for i in range(1, n)}  # route index (same as customer initially)
        route_start = {i: i for i in range(1, n)}  # first customer after depot
        route_end = {i: i for i in range(1, n)}   # last customer before depot
        # We'll maintain routes as list of lists
        route_list = {i: [0, i, 0] for i in range(1, n)}
        merger_counter = 0
        for s, i, j in savings:
            if len(routes) <= truck_count:
                break
            # Randomization: with probability 0.2, skip best and choose random among top 10%
            if random.random() < 0.2:
                # find a random savings among top 10%
                top_count = max(1, len(savings)//10)
                idx = random.randint(0, top_count-1)
                # but we processed linearly; we can skip to a random later savings? It's messy. Simpler: we'll just proceed linearly but with chance of skipping merge.
                # Actually, let's just do a probabilistic merge: only merge with probability 0.8 even if feasible.
                pass
            # feasibility: merge i and j if they are in different routes and their free endpoints match
            ri = cust_to_route[i]
            rj = cust_to_route[j]
            if ri == rj:
                continue
            # Check if we can connect i's end to j's start or vice versa, respecting orientation
            # We need that i is at an end of its route and j is at an end of its route.
            # Free endpoints: route_start[ri] and route_end[ri] (cust at ends) 
            # Actually we track the first and last customer after depot
            ends_i = (route_start[ri], route_end[ri])
            ends_j = (route_start[rj], route_end[rj])
            # Possible connections: (i_end, j_start) or (j_end, i_start) etc.
            # We want to merge such that the savings are realized: connecting i to j or j to i.
            # Since savings is symmetric, we can connect either end.
            # Determine if i is at end of its route and j at end of its route
            # We need to know if i == route_end[ri] or i == route_start[ri]; similarly j
            if i == route_end[ri] and j == route_start[rj]:
                # merge by connecting route_end[ri] to route_start[rj]
                new_route = route_list[ri] + route_list[rj][1:]  # remove depot at start of rj
                # update
                new_start = route_start[ri]
                new_end = route_end[rj]
                # merge into ri, remove rj
                route_list[ri] = new_route
                route_start[ri] = new_start
                route_end[ri] = new_end
                # update cust_to_route for customers in rj
                for c in route_list[rj][1:-1]:
                    cust_to_route[c] = ri
                del route_list[rj]
                del route_start[rj]
                del route_end[rj]
            elif i == route_start[ri] and j == route_end[rj]:
                new_route = route_list[rj] + route_list[ri][1:]
                new_start = route_start[rj]
                new_end = route_end[ri]
                route_list[rj] = new_route
                route_start[rj] = new_start
                route_end[rj] = new_end
                for c in route_list[ri][1:-1]:
                    cust_to_route[c] = rj
                del route_list[ri]
                del route_start[ri]
                del route_end[ri]
            elif i == route_end[ri] and j == route_end[rj]:
                # reverse rj and connect
                reversed_rj = [0] + list(reversed(route_list[rj][1:-1])) + [0]
                new_route = route_list[ri] + reversed_rj[1:]
                new_end = route_start[rj]  # after reversing, original start becomes end
                route_list[ri] = new_route
                route_end[ri] = new_end
                for c in route_list[rj][1:-1]:
                    cust_to_route[c] = ri
                del route_list[rj]
                del route_start[rj]
                del route_end[rj]
            elif i == route_start[ri] and j == route_start[rj]:
                # reverse ri and connect
                reversed_ri = [0] + list(reversed(route_list[ri][1:-1])) + [0]
                new_route = reversed_ri + route_list[rj][1:]
                new_start = route_end[ri]
                route_list[rj] = new_route
                route_start[rj] = new_start
                for c in route_list[ri][1:-1]:
                    cust_to_route[c] = rj
                del route_list[ri]
                del route_start[ri]
                del route_end[ri]
            else:
                continue
            # After merge, check route count
            if len(route_list) <= truck_count:
                break
            merger_counter += 1
        # Now we have routes in route_list values
        routes = list(route_list.values())
        # If we have fewer than truck_count, add empty routes
        while len(routes) < truck_count:
            routes.append([0,0])
        # If we have more than truck_count, we need to split? Usually Clark-Wright gives exactly truck_count if we stop early. But here we might have too many if no merges. For safety, we'll pick the first truck_count routes? Not good. Better: after initial construction, we might have more routes than truck_count. We'll need to assign remaining customers to existing routes via cheapest insertion. Simpler: we'll ensure we stop merging when we reach exactly truck_count. But if we have failures, we might have more. So let's do a post-processing: if len(routes) > truck_count, we'll merge the smallest routes into others using cheapest insertion until we have truck_count.
        # For brevity, I'll implement a quick cheapest insertion to reduce routes.
        while len(routes) > truck_count:
            # find smallest route (by number of customers) to break
            min_len = float('inf')
            min_idx = -1
            for idx, r in enumerate(routes):
                if len(r)-2 < min_len:
                    min_len = len(r)-2
                    min_idx = idx
            # break that route: insert its customers into other routes
            broken_route = routes.pop(min_idx)
            for cust in broken_route[1:-1]:
                best_inc = float('inf')
                best_route = -1
                best_pos = -1
                for r_idx, r in enumerate(routes):
                    for pos in range(1, len(r)):
                        inc = distance_matrix[r[pos-1], cust] + distance_matrix[cust, r[pos]] - distance_matrix[r[pos-1], r[pos]]
                        if inc < best_inc:
                            best_inc = inc
                            best_route = r_idx
                            best_pos = pos
                routes[best_route] = routes[best_route][:best_pos] + [cust] + routes[best_route][best_pos:]

        # 2-opt improvement
        def two_opt(route, max_iter=10):
            route = list(route)
            improved = True
            it = 0
            while improved and it < max_iter:
                improved = False
                it += 1
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if route_dist(new_route) < route_dist(route):
                            route = new_route
                            improved = True
            return route
        for idx in range(len(routes)):
            if len(routes[idx]) > 2:
                routes[idx] = two_opt(routes[idx], max_iter=10)

        # Balancing relocation
        lengths = [route_dist(r) for r in routes]
        improved = True
        bal_iter = 0
        max_bal = n * truck_count
        while improved and bal_iter < max_bal:
            improved = False
            bal_iter += 1
            max_len_idx = max(range(len(lengths)), key=lambda i: lengths[i])
            min_len_idx = min(range(len(lengths)), key=lambda i: lengths[i])
            if max_len_idx == min_len_idx:
                break
            max_route = routes[max_len_idx]
            best_move = None
            best_reduction = 0
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max = max_route[:pos] + max_route[pos+1:]
                new_max_len = route_dist(new_max)
                min_route = routes[min_len_idx]
                best_pos = -1
                best_min_len = float('inf')
                for p in range(1, len(min_route)):
                    new_min = min_route[:p] + [cust] + min_route[p:]
                    l = route_dist(new_min)
                    if l < best_min_len:
                        best_min_len = l
                        best_pos = p
                other_lengths = [lengths[i] for i in range(len(lengths)) if i not in (max_len_idx, min_len_idx)]
                new_global_max = max(new_max_len, best_min_len, max(other_lengths) if other_lengths else 0)
                old_global_max = lengths[max_len_idx]
                reduction = old_global_max - new_global_max
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_move = (cust, best_pos, new_max, min_route[:best_pos] + [cust] + min_route[best_pos:])
            if best_move is not None and best_reduction > 0:
                cust, pos, new_max_route, new_min_route = best_move
                routes[max_len_idx] = new_max_route
                routes[min_len_idx] = new_min_route
                lengths[max_len_idx] = route_dist(new_max_route)
                lengths[min_len_idx] = route_dist(new_min_route)
                improved = True
                if max(lengths) < best_max:
                    best_max = max(lengths)
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
        
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
    
    # Final best
    return best_routes