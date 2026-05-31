import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    # Cheapest insertion construction
    routes = [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    for cust in customers:
        best_inc = float('inf')
        best_route = -1
        best_pos = -1
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                inc = (distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] -
                       distance_matrix[route[pos-1], route[pos]])
                if inc < best_inc or (inc == best_inc and r_idx < best_route):
                    best_inc = inc
                    best_route = r_idx
                    best_pos = pos
        route = routes[best_route]
        routes[best_route] = route[:best_pos] + [cust] + route[best_pos:]
    report_best_vrp(routes)
    
    best_routes = [r[:] for r in routes]
    best_max = max(route_distance(r) for r in routes)
    
    max_iter = n * truck_count * 10
    tabu_list = {}
    tenure = 5
    for it in range(max_iter):
        # Build customer location map
        cust_to_route = {}
        for r_idx, route in enumerate(routes):
            for pos, c in enumerate(route):
                if c != 0:
                    cust_to_route[c] = (r_idx, pos)
        
        best_move = None
        best_new_max = float('inf')
        best_new_total = float('inf')
        best_primary = None
        best_secondary = None
        
        # Relocate moves
        for cust, (cur_route_idx, cur_pos) in cust_to_route.items():
            new_src = routes[cur_route_idx][:cur_pos] + routes[cur_route_idx][cur_pos+1:]
            src_dist = route_distance(new_src)
            for dst_route_idx, dst_route in enumerate(routes):
                for ins_pos in range(1, len(dst_route)):
                    if dst_route_idx == cur_route_idx and ins_pos == cur_pos:
                        continue
                    new_dst = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                    dst_dist = route_distance(new_dst)
                    new_max = src_dist if src_dist > dst_dist else dst_dist
                    # Compute other route distances
                    total = src_dist + dst_dist
                    for r_idx_other, r_other in enumerate(routes):
                        if r_idx_other == cur_route_idx or r_idx_other == dst_route_idx:
                            continue
                        other_dist = route_distance(r_other)
                        total += other_dist
                        if other_dist > new_max:
                            new_max = other_dist
                    # Check tabu
                    is_tabu = (cust, cur_route_idx) in tabu_list and tabu_list[(cust, cur_route_idx)] > 0
                    if is_tabu and new_max >= best_max:
                        continue
                    # Aspiration
                    if new_max < best_max:
                        is_tabu = False
                    # Compare
                    key = (new_max, new_total, cust, cur_route_idx)
                    if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total) or \
                       (new_max == best_new_max and new_total == best_new_total and cust < best_primary) or \
                       (new_max == best_new_max and new_total == best_new_total and cust == best_primary and cur_route_idx < best_secondary):
                        best_new_max = new_max
                        best_new_total = new_total
                        best_primary = cust
                        best_secondary = cur_route_idx
                        best_move = ('relocate', cust, cur_route_idx, dst_route_idx, ins_pos, new_src, new_dst)
        
        # Swap moves
        customers_list = list(range(1, n))
        for i in range(len(customers_list)):
            cust1 = customers_list[i]
            for j in range(i+1, len(customers_list)):
                cust2 = customers_list[j]
                # Find routes and positions
                r1, p1 = cust_to_route[cust1]
                r2, p2 = cust_to_route[cust2]
                if r1 == r2:
                    # Same route: swap positions
                    route = routes[r1][:]
                    p1 = route.index(cust1)  # ensure correct after any changes? but we have mapping
                    p2 = route.index(cust2)
                    # Swap
                    new_route = route[:p1] + [route[p2]] + route[p1+1:p2] + [route[p1]] + route[p2+1:]
                    new_dists = {r1: route_distance(new_route)}
                    new_max = new_dists[r1]
                    total = new_dists[r1]
                    for r_idx, route in enumerate(routes):
                        if r_idx != r1:
                            d = route_distance(route)
                            new_dists[r_idx] = d
                            total += d
                            if d > new_max:
                                new_max = d
                    new_routes = routes[:]
                    new_routes[r1] = new_route
                else:
                    # Different routes: swap customers
                    route1 = routes[r1][:]
                    route2 = routes[r2][:]
                    # Remove cust1 from route1, cust2 from route2
                    route1_no_c1 = route1[:p1] + route1[p1+1:]
                    route2_no_c2 = route2[:p2] + route2[p2+1:]
                    # Insert cust1 into route2 at its original position (same position) or best? Keep original position
                    # To keep it simple, we insert at the position where cust2 was
                    new_route1 = route1_no_c1
                    new_route2 = route2_no_c2[:p2] + [cust1] + route2_no_c2[p2:]
                    # But we need to ensure order: we could insert at same index or adjust. Let's use insertion at p2 after removal
                    # Actually after removal, p2 might be out of bounds if cust2 was not at p2? We have mapping correct.
                    # Simpler: construct by swapping values
                    new_route1 = routes[r1][:]
                    new_route2 = routes[r2][:]
                    # Replace cust1 with cust2 in route1, and cust2 with cust1 in route2
                    # But positions may differ. We'll do: remove cust1 from route1, insert cust2 at the position where cust1 was? That's a relocate. For swap we want to exchange. So we remove both and insert each in other's route at the other's position.
                    # Let's do correctly:
                    pos1 = new_route1.index(cust1)
                    pos2 = new_route2.index(cust2)
                    new_route1[pos1] = cust2
                    new_route2[pos2] = cust1
                    # After replacement, check if routes still valid (no duplicates). Since we replaced, it's fine.
                    new_dists = {r1: route_distance(new_route1), r2: route_distance(new_route2)}
                    new_max = max(new_dists[r1], new_dists[r2])
                    total = route_distance(new_route1) + route_distance(new_route2)
                    for r_idx, route in enumerate(routes):
                        if r_idx != r1 and r_idx != r2:
                            d = route_distance(route)
                            new_dists[r_idx] = d
                            total += d
                            if d > new_max:
                                new_max = d
                    new_routes = routes[:]
                    new_routes[r1] = new_route1
                    new_routes[r2] = new_route2
                # Check tabu for swap
                tabu_key = (min(cust1, cust2), max(cust1, cust2))
                is_tabu = tabu_key in tabu_list and tabu_list[tabu_key] > 0
                if is_tabu and new_max >= best_max:
                    continue
                if new_max < best_max:
                    is_tabu = False
                # Compare
                primary = min(cust1, cust2)
                secondary = max(cust1, cust2)
                if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total) or \
                   (new_max == best_new_max and new_total == best_new_total and primary < best_primary) or \
                   (new_max == best_new_max and new_total == best_new_total and primary == best_primary and secondary < best_secondary):
                    best_new_max = new_max
                    best_new_total = new_total
                    best_primary = primary
                    best_secondary = secondary
                    best_move = ('swap', cust1, cust2, r1, r2, new_routes)
        
        if best_move is None:
            break
        # Apply best move
        if best_move[0] == 'relocate':
            _, cust, src_route_idx, dst_route_idx, ins_pos, new_src, new_dst = best_move
            if src_route_idx == dst_route_idx:
                routes[src_route_idx] = new_dst
            else:
                routes[src_route_idx] = new_src
                routes[dst_route_idx] = new_dst
            tabu_list[(cust, src_route_idx)] = tenure + 1
        else:
            _, cust1, cust2, r1, r2, new_routes = best_move
            routes = new_routes
            tabu_list[(min(cust1,cust2), max(cust1,cust2))] = tenure + 1
        # Decrement tabu
        for key in list(tabu_list.keys()):
            tabu_list[key] -= 1
            if tabu_list[key] <= 0:
                del tabu_list[key]
        # Update best solution
        current_max = max(route_distance(r) for r in routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
    return best_routes