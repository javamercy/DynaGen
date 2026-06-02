import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # initial routes: each customer as a separate route (including depot)
    routes = [[0, c, 0] for c in customers]
    # track for each route its first and last customer (excluding depot)
    first = {i: c for i, c in enumerate(customers)}
    last = {i: c for i, c in enumerate(customers)}
    route_count = len(routes)
    
    # compute savings for all pairs of customers (i < j)
    savings = []
    for i in range(len(customers)):
        for j in range(i+1, len(customers)):
            ci = customers[i]
            cj = customers[j]
            s = distance_matrix[0][ci] + distance_matrix[0][cj] - distance_matrix[ci][cj]
            savings.append((s, ci, cj))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))  # descending savings, tie by indeces
    
    # mapping from customer to route index
    cust_to_route = {c: i for i, c in enumerate(customers)}
    
    # merge until we have exactly truck_count routes
    used = [False] * len(savings)
    idx = 0
    while route_count > truck_count and idx < len(savings):
        s, ci, cj = savings[idx]
        idx += 1
        ri = cust_to_route[ci]
        rj = cust_to_route[cj]
        if ri == rj:
            continue
        # check if ci is an endpoint of its route and cj is an endpoint of its route
        route_i = routes[ri]
        route_j = routes[rj]
        # endpoints: first[ri] is first customer after 0, last[ri] is last before 0
        if (ci == first[ri] or ci == last[ri]) and (cj == first[rj] or cj == last[rj]):
            # possible merges: we need to connect without crossing
            # We'll try to connect last of ri to first of rj, or first of ri to last of rj (reverse one route)
            # But to keep orientation, we can simply concatenate in one order
            # We'll choose the merge that yields positive saving (should be for this pair)
            # If both endpoints match, we can merge by appending one route to the other
            # We'll decide based on orientation to avoid reversing (but reversing is okay)
            if ci == last[ri] and cj == first[rj]:
                # merge ri then rj
                new_route = route_i[:-1] + route_j[1:]  # remove overlapping 0
                new_first = first[ri]
                new_last = last[rj]
            elif ci == first[ri] and cj == last[rj]:
                # merge rj then ri (reverse order? Actually we can simply concatenate with reversed ri? Let's do something simpler: if first[ri] equals last[rj]? No.
                # We can merge rj then ri: route_j[:-1] + route_i[1:]
                new_route = route_j[:-1] + route_i[1:]
                new_first = first[rj]
                new_last = last[ri]
            elif ci == last[ri] and cj == last[rj]:
                # need to reverse rj? Or we can connect by reversing rj so that its first becomes last? We'll just reverse rj's internal order (excluding depot)
                # For simplicity, we'll only merge in the standard two ways. If not, skip.
                continue
            elif ci == first[ri] and cj == first[rj]:
                # need to reverse ri or rj
                continue
            else:
                continue
            # update routes: remove both old routes, add new route
            # keep track of indices: we will replace ri with new route, and remove rj
            # first update cust_to_route for customers in route_j to new index
            for c in route_j[1:-1]:
                cust_to_route[c] = ri
            # also for route_i (already same index)
            routes[ri] = new_route
            # remove route rj (pop and adjust indices)
            routes.pop(rj)
            # remove rj from tracking
            del first[rj]
            del last[rj]
            # update first and last for ri
            first[ri] = new_first
            last[ri] = new_last
            # adjust cust_to_route for customers that were in route_i (already correct) and route_j (already updated)
            # Also decrement route_count
            route_count -= 1
            # adjust indices in first/last for routes after rj
            # Since we popped rj, indices of later routes shift down by 1
            # Rebuild first/last for indices > rj
            keys = sorted(first.keys())
            for k in keys:
                if k > rj:
                    first[k-1] = first.pop(k)
                    last[k-1] = last.pop(k)
            # also cust_to_route for customers in those routes
            for k in range(rj, len(routes)):
                for c in routes[k][1:-1]:
                    cust_to_route[c] = k
            # we must also update the mapping for the route that was merged into, but it already has index ri (which is < rj or >? Popping affects indices)
            # After popping rj, if ri > rj, then ri decreases by 1. Let's handle below.
            # Instead of complex index updates, simpler approach: after each merge, rebuild cust_to_route from scratch? But that could be O(n^2). Since n is bounded, it's okay.
    
    # After merging, ensure exactly truck_count routes
    # If less than truck_count, add empty routes
    while len(routes) < truck_count:
        routes.append([0, 0])
    # If more than truck_count (should not happen), we may need to merge further? But we already looped to reduce to truck_count.
    
    # Simple improvement: intra-route 2-opt
    for r in range(len(routes)):
        route = routes[r]
        if len(route) <= 3:
            continue
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j-i == 1:
                        continue
                    # compute distance change if we reverse segment i..j
                    old_dist = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                    new_dist = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                    if new_dist < old_dist:
                        # reverse segment
                        route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                routes[r] = route
    
    # Inter-route improvement: minimize max distance
    def route_distance(route):
        total = 0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total
    
    # compute initial max
    best_routes = [list(r) for r in routes]
    best_max = max(route_distance(r) for r in routes)
    # call report_best_vrp
    try:
        report_best_vrp(best_routes)
    except:
        pass
    
    # try relocate moves
    improved = True
    max_iter = n * n  # bounded
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        # find longest route
        max_dist = 0
        max_idx = 0
        for i, r in enumerate(routes):
            d = route_distance(r)
            if d > max_dist:
                max_dist = d
                max_idx = i
        # try moving a customer from longest route to another route
        route_from = routes[max_idx]
        for cust in route_from[1:-1]:  # exclude depot
            for r_to_idx in range(len(routes)):
                if r_to_idx == max_idx:
                    continue
                route_to = routes[r_to_idx]
                # try inserting at every position in route_to
                for pos in range(1, len(route_to)):
                    new_route_to = route_to[:pos] + [cust] + route_to[pos:]
                    new_route_from = [x for x in route_from if x != cust or x == 0]  # remove cust, keep depots
                    # ensure new_route_from has at least depots
                    if len(new_route_from) == 1:  # only depot
                        new_route_from = [0, 0]
                    # compute new max
                    new_max = max(route_distance(new_route_from), route_distance(new_route_to), max(route_distance(r) for i, r in enumerate(routes) if i != max_idx and i != r_to_idx))
                    if new_max < best_max:
                        # accept
                        routes[max_idx] = new_route_from
                        routes[r_to_idx] = new_route_to
                        best_routes = [list(r) for r in routes]
                        best_max = new_max
                        improved = True
                        try:
                            report_best_vrp(best_routes)
                        except:
                            pass
                        break
                if improved:
                    break
            if improved:
                break
        # also try swap between longest and other routes
        if not improved:
            for r_to_idx in range(len(routes)):
                if r_to_idx == max_idx:
                    continue
                route_to = routes[r_to_idx]
                for cust_from in route_from[1:-1]:
                    for cust_to in route_to[1:-1]:
                        if cust_from == cust_to:
                            continue
                        # swap
                        new_route_from = [0] + [cust_to if x==cust_from else x for x in route_from[1:-1]] + [0]
                        new_route_to = [0] + [cust_from if x==cust_to else x for x in route_to[1:-1]] + [0]
                        new_max = max(route_distance(new_route_from), route_distance(new_route_to), max(route_distance(r) for i, r in enumerate(routes) if i != max_idx and i != r_to_idx))
                        if new_max < best_max:
                            routes[max_idx] = new_route_from
                            routes[r_to_idx] = new_route_to
                            best_routes = [list(r) for r in routes]
                            best_max = new_max
                            improved = True
                            try:
                                report_best_vrp(best_routes)
                            except:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
    return best_routes