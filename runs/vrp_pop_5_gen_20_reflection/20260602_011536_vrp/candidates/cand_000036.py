import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix.tolist()
    random.seed(0)
    
    def route_distance(route):
        if len(route) < 2:
            return 0
        return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))
    
    def objective(routes):
        return max(route_distance(r) for r in routes)
    
    # Initial minimax construction
    routes = [[0,0] for _ in range(truck_count)]
    unassigned = list(range(1,n))
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
                    new_dist = 0
                    prev = route[0]
                    for k in range(1, len(route)):
                        if k == pos:
                            new_dist += dist[prev][node]
                            prev = node
                        new_dist += dist[prev][route[k]]
                        prev = route[k]
                    current_max = 0
                    for rr in range(truck_count):
                        if rr == r:
                            d = new_dist
                        else:
                            d = route_distance(routes[rr])
                        if d > current_max:
                            current_max = d
                    if current_max < best_max or (current_max == best_max and new_dist < best_total):
                        best_max = current_max
                        best_total = new_dist
                        best_node = node
                        best_route = r
                        best_pos = pos
        routes[best_route].insert(best_pos, best_node)
        unassigned.remove(best_node)
    
    best_routes = [list(r) for r in routes]
    best_obj = objective(best_routes)
    report_best_vrp(best_routes)
    current_routes = [list(r) for r in routes]
    current_obj = best_obj
    
    # VNS parameters
    max_iter = min(50, 2*n)
    neighborhoods = ['relocate', 'swap', '2opt']
    
    for it in range(max_iter):
        # Shaking: generate random moves
        new_routes = [list(r) for r in current_routes]
        shake_intensity = random.randint(1, max(1, n//10))
        for _ in range(shake_intensity):
            nh = random.choice(neighborhoods)
            if nh == 'relocate':
                # relocate a random customer to a random position in a random route
                customers = [c for route in new_routes for c in route[1:-1]]
                if not customers:
                    continue
                node = random.choice(customers)
                # find which route currently has node
                src_route_idx = None
                for idx, route in enumerate(new_routes):
                    if node in route:
                        src_route_idx = idx
                        break
                if src_route_idx is None:
                    continue
                src_route = new_routes[src_route_idx]
                # remove node
                pos = src_route.index(node)
                new_src = src_route[:pos] + src_route[pos+1:]
                if len(new_src) < 2:
                    new_src = [0,0]
                new_routes[src_route_idx] = new_src
                # choose target route and position
                tgt_route_idx = random.randrange(truck_count)
                tgt_route = new_routes[tgt_route_idx]
                if len(tgt_route) < 2:
                    tgt_route = [0,0]
                insert_pos = random.randint(1, len(tgt_route)-1)
                tgt_route.insert(insert_pos, node)
                new_routes[tgt_route_idx] = tgt_route
            elif nh == 'swap':
                # swap two random customers from different routes
                routes_with_two = [i for i, r in enumerate(new_routes) if len(r) > 2]
                if len(routes_with_two) < 2:
                    continue
                r1 = random.choice(routes_with_two)
                r2 = random.choice([r for r in routes_with_two if r != r1])
                route1 = new_routes[r1]
                route2 = new_routes[r2]
                # choose one customer from each (excluding depots)
                c1 = random.choice(route1[1:-1])
                c2 = random.choice(route2[1:-1])
                # replace
                pos1 = route1.index(c1)
                pos2 = route2.index(c2)
                route1[pos1] = c2
                route2[pos2] = c1
                new_routes[r1] = route1
                new_routes[r2] = route2
            elif nh == '2opt':
                # apply random 2-opt on a random route
                r_idx = random.randrange(truck_count)
                route = new_routes[r_idx]
                if len(route) <= 3:
                    continue
                i = random.randint(1, len(route)-3)
                j = random.randint(i+1, len(route)-2)
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_routes[r_idx] = new_route
        
        # Local search (best improvement on max route distance)
        improved = True
        local_iter = 0
        while improved and local_iter < 10:
            improved = False
            # iterate over neighborhoods in random order for diversification
            random.shuffle(neighborhoods)
            for nh in neighborhoods:
                if nh == 'relocate':
                    # best improvement relocate
                    best_move = None
                    best_new_obj = float('inf')
                    for src_idx, src_route in enumerate(new_routes):
                        if len(src_route) <= 2:
                            continue
                        for pos_src in range(1, len(src_route)-1):
                            node = src_route[pos_src]
                            # remove node
                            temp_route = src_route[:pos_src] + src_route[pos_src+1:]
                            if len(temp_route) < 2:
                                temp_route = [0,0]
                            for tgt_idx in range(truck_count):
                                tgt_route = list(new_routes[tgt_idx])
                                for pos_tgt in range(1, len(tgt_route)):
                                    # insert node at pos_tgt
                                    new_tgt = tgt_route[:pos_tgt] + [node] + tgt_route[pos_tgt:]
                                    # compute new routes array
                                    test_routes = [list(r) for r in new_routes]
                                    test_routes[src_idx] = temp_route
                                    test_routes[tgt_idx] = new_tgt
                                    obj = objective(test_routes)
                                    if obj < best_new_obj:
                                        best_new_obj = obj
                                        best_move = (src_idx, pos_src, tgt_idx, pos_tgt)
                    if best_move and best_new_obj < current_obj:
                        src_idx, pos_src, tgt_idx, pos_tgt = best_move
                        # apply move
                        src_route = new_routes[src_idx]
                        node = src_route.pop(pos_src)
                        if len(src_route) < 2:
                            src_route = [0,0]
                        new_routes[src_idx] = src_route
                        tgt_route = new_routes[tgt_idx]
                        tgt_route.insert(pos_tgt, node)
                        new_routes[tgt_idx] = tgt_route
                        current_obj = best_new_obj
                        improved = True
                        break  # restart neighborhood cycle
                elif nh == 'swap':
                    best_move = None
                    best_new_obj = float('inf')
                    for i in range(truck_count):
                        for j in range(i+1, truck_count):
                            route_i = new_routes[i]
                            route_j = new_routes[j]
                            if len(route_i) < 3 or len(route_j) < 3:
                                continue
                            # iterate over customer positions in each
                            for pos_i in range(1, len(route_i)-1):
                                for pos_j in range(1, len(route_j)-1):
                                    cust_i = route_i[pos_i]
                                    cust_j = route_j[pos_j]
                                    # swap
                                    test_i = list(route_i)
                                    test_j = list(route_j)
                                    test_i[pos_i] = cust_j
                                    test_j[pos_j] = cust_i
                                    test_routes = [list(r) for r in new_routes]
                                    test_routes[i] = test_i
                                    test_routes[j] = test_j
                                    obj = objective(test_routes)
                                    if obj < best_new_obj:
                                        best_new_obj = obj
                                        best_move = (i, pos_i, j, pos_j)
                    if best_move and best_new_obj < current_obj:
                        i, pos_i, j, pos_j = best_move
                        route_i = new_routes[i]
                        route_j = new_routes[j]
                        cust_i = route_i[pos_i]
                        cust_j = route_j[pos_j]
                        route_i[pos_i] = cust_j
                        route_j[pos_j] = cust_i
                        new_routes[i] = route_i
                        new_routes[j] = route_j
                        current_obj = best_new_obj
                        improved = True
                        break
                elif nh == '2opt':
                    best_move = None
                    best_new_obj = float('inf')
                    for r_idx in range(truck_count):
                        route = new_routes[r_idx]
                        if len(route) <= 3:
                            continue
                        for i in range(1, len(route)-2):
                            for j in range(i+1, len(route)-1):
                                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                                test_routes = [list(r) for r in new_routes]
                                test_routes[r_idx] = new_route
                                obj = objective(test_routes)
                                if obj < best_new_obj:
                                    best_new_obj = obj
                                    best_move = (r_idx, i, j)
                    if best_move and best_new_obj < current_obj:
                        r_idx, i, j = best_move
                        route = new_routes[r_idx]
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_routes[r_idx] = new_route
                        current_obj = best_new_obj
                        improved = True
                        break
            local_iter += 1
        
        # Acceptance: only if better than current
        if current_obj < best_obj:
            best_obj = current_obj
            best_routes = [list(r) for r in new_routes]
            report_best_vrp(best_routes)
        # else keep current solution (no uphill)
    
    return best_routes