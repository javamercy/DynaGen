import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def compute_max_dist(routes):
        return max(route_distance(r) for r in routes)
    
    def construct_routes(tie_break_random=False):
        routes = [[0, 0] for _ in range(truck_count)]
        route_distances = [0.0 for _ in range(truck_count)]
        unassigned = set(customers)
        while unassigned:
            best_custs = []
            best_regret = -1
            best_route_idx = None
            best_pos = None
            best_new_dist = None
            for cust in list(unassigned):
                costs = []
                for r in range(truck_count):
                    route = routes[r]
                    for pos in range(1, len(route)):
                        delta = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        new_dist = route_distances[r] + delta
                        costs.append((new_dist, r, pos))
                costs.sort(key=lambda x: x[0])
                best = costs[0]
                second = costs[1] if len(costs) > 1 else (float('inf'), -1, -1)
                regret = second[0] - best[0]
                if tie_break_random:
                    if regret > best_regret:
                        best_regret = regret
                        best_custs = [(cust, best[1], best[2], best[0])]
                    elif regret == best_regret:
                        best_custs.append((cust, best[1], best[2], best[0]))
                else:
                    if regret > best_regret:
                        best_regret = regret
                        best_custs = [(cust, best[1], best[2], best[0])]
                    elif regret == best_regret:
                        best_custs.append((cust, best[1], best[2], best[0]))
            if len(best_custs) > 1:
                chosen = random.choice(best_custs)
            else:
                chosen = best_custs[0]
            best_cust, best_route_idx, best_pos, best_new_dist = chosen
            routes[best_route_idx].insert(best_pos, best_cust)
            route_distances[best_route_idx] = best_new_dist
            unassigned.remove(best_cust)
        return routes, route_distances
    
    routes, route_distances = construct_routes(tie_break_random=False)
    best_routes = [r[:] for r in routes]
    best_max = compute_max_dist(routes)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass
    
    current_routes = [r[:] for r in routes]
    current_max = best_max
    deviation = 0.1 * current_max
    max_iter = 600
    no_improve = 0
    for it in range(max_iter):
        improved = False
        
        # Targeted relocate perturbation when stagnation
        if no_improve >= 20:
            # Find longest route
            longest_idx = max(range(truck_count), key=lambda i: route_distances[i])
            # Find a shorter route (any other route with distance less than longest's)
            shorter_indices = [i for i in range(truck_count) if i != longest_idx and route_distances[i] < route_distances[longest_idx]]
            if shorter_indices and len(current_routes[longest_idx]) > 2:
                shorter_idx = random.choice(shorter_indices)
                # Pick a random customer from longest route (excluding depots)
                customers_longest = current_routes[longest_idx][1:-1]
                if customers_longest:
                    cust = random.choice(customers_longest)
                    # Remove cust from longest route
                    pos_longest = current_routes[longest_idx].index(cust)
                    current_routes[longest_idx].pop(pos_longest)
                    # Insert into shorter route at random position (excluding depots)
                    insert_pos = random.randint(1, len(current_routes[shorter_idx])-1)
                    current_routes[shorter_idx].insert(insert_pos, cust)
                    # Update route distances
                    route_distances[longest_idx] = route_distance(current_routes[longest_idx])
                    route_distances[shorter_idx] = route_distance(current_routes[shorter_idx])
                    current_max = compute_max_dist(current_routes)
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in current_routes]
                        try:
                            report_best_vrp(best_routes)
                        except NameError:
                            pass
            no_improve = 0
            improved = True
        
        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
            best_i = None
            best_j = None
            best_new_max = current_max + deviation + 1
            for i in range(1, len(route)-2):
                for j in range(i+2, len(route)-1):
                    new_route = route[:i+1] + route[j:i:-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    candidate_max = new_dist
                    for rr in range(truck_count):
                        if rr != r_idx:
                            candidate_max = max(candidate_max, route_distances[rr])
                    if candidate_max < best_new_max:
                        best_new_max = candidate_max
                        best_i = i
                        best_j = j
            if best_i is not None and best_new_max <= current_max + deviation:
                new_route = current_routes[r_idx][:best_i+1] + current_routes[r_idx][best_j:best_i:-1] + current_routes[r_idx][best_j+1:]
                current_routes[r_idx] = new_route
                route_distances[r_idx] = route_distance(new_route)
                current_max = compute_max_dist(current_routes)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in current_routes]
                    try:
                        report_best_vrp(best_routes)
                    except NameError:
                        pass
                improved = True
        
        # Inter-route relocate
        best_move = None
        best_new_max = current_max + deviation + 1
        for r_from in range(truck_count):
            if len(current_routes[r_from]) <= 2:
                continue
            for pos_from in range(1, len(current_routes[r_from])-1):
                cust = current_routes[r_from][pos_from]
                prev = current_routes[r_from][pos_from-1]
                next_c = current_routes[r_from][pos_from+1]
                delta_from = distance_matrix[prev, next_c] - distance_matrix[prev, cust] - distance_matrix[cust, next_c]
                new_from_dist = route_distances[r_from] + delta_from
                for r_to in range(truck_count):
                    if r_to == r_from:
                        continue
                    route_to = current_routes[r_to]
                    for pos_to in range(1, len(route_to)):
                        prev_to = route_to[pos_to-1]
                        next_to = route_to[pos_to]
                        delta_to = distance_matrix[prev_to, cust] + distance_matrix[cust, next_to] - distance_matrix[prev_to, next_to]
                        new_to_dist = route_distances[r_to] + delta_to
                        candidate_max = max(new_from_dist, new_to_dist)
                        for rr in range(truck_count):
                            if rr != r_from and rr != r_to:
                                candidate_max = max(candidate_max, route_distances[rr])
                        if candidate_max < best_new_max:
                            best_new_max = candidate_max
                            best_move = (r_from, pos_from, r_to, pos_to, new_from_dist, new_to_dist)
        if best_move is not None and best_new_max <= current_max + deviation:
            r_from, pos_from, r_to, pos_to, new_from_dist, new_to_dist = best_move
            cust = current_routes[r_from].pop(pos_from)
            current_routes[r_to].insert(pos_to, cust)
            route_distances[r_from] = new_from_dist
            route_distances[r_to] = new_to_dist
            current_max = compute_max_dist(current_routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in current_routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
            improved = True
        
        # Inter-route swap
        best_move = None
        best_new_max = current_max + deviation + 1
        for r1 in range(truck_count):
            if len(current_routes[r1]) <= 2:
                continue
            for pos1 in range(1, len(current_routes[r1])-1):
                cust1 = current_routes[r1][pos1]
                prev1 = current_routes[r1][pos1-1]
                next1 = current_routes[r1][pos1+1]
                for r2 in range(r1+1, truck_count):
                    if len(current_routes[r2]) <= 2:
                        continue
                    for pos2 in range(1, len(current_routes[r2])-1):
                        cust2 = current_routes[r2][pos2]
                        prev2 = current_routes[r2][pos2-1]
                        next2 = current_routes[r2][pos2+1]
                        delta1 = distance_matrix[prev1, cust2] + distance_matrix[cust2, next1] - distance_matrix[prev1, cust1] - distance_matrix[cust1, next1]
                        new_dist1 = route_distances[r1] + delta1
                        delta2 = distance_matrix[prev2, cust1] + distance_matrix[cust1, next2] - distance_matrix[prev2, cust2] - distance_matrix[cust2, next2]
                        new_dist2 = route_distances[r2] + delta2
                        candidate_max = max(new_dist1, new_dist2)
                        for rr in range(truck_count):
                            if rr != r1 and rr != r2:
                                candidate_max = max(candidate_max, route_distances[rr])
                        if candidate_max < best_new_max:
                            best_new_max = candidate_max
                            best_move = (r1, pos1, r2, pos2, new_dist1, new_dist2)
        if best_move is not None and best_new_max <= current_max + deviation:
            r1, pos1, r2, pos2, new_dist1, new_dist2 = best_move
            cust1 = current_routes[r1][pos1]
            cust2 = current_routes[r2][pos2]
            current_routes[r1][pos1] = cust2
            current_routes[r2][pos2] = cust1
            route_distances[r1] = new_dist1
            route_distances[r2] = new_dist2
            current_max = compute_max_dist(current_routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in current_routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
            improved = True
        
        # Inter-route cross-exchange
        best_move = None
        best_new_max = current_max + deviation + 1
        for r1 in range(truck_count):
            for r2 in range(r1+1, truck_count):
                route1 = current_routes[r1]
                route2 = current_routes[r2]
                if len(route1) <= 2 or len(route2) <= 2:
                    continue
                for i in range(1, len(route1)-2):
                    for j in range(1, len(route2)-2):
                        new_route1 = route1[:i+1] + route2[j+1:]
                        new_route2 = route2[:j+1] + route1[i+1:]
                        new_dist1 = route_distance(new_route1)
                        new_dist2 = route_distance(new_route2)
                        candidate_max = max(new_dist1, new_dist2)
                        for rr in range(truck_count):
                            if rr != r1 and rr != r2:
                                candidate_max = max(candidate_max, route_distances[rr])
                        if candidate_max < best_new_max:
                            best_new_max = candidate_max
                            best_move = (r1, r2, i, j, new_route1, new_route2, new_dist1, new_dist2)
        if best_move is not None and best_new_max <= current_max + deviation:
            r1, r2, i, j, new_route1, new_route2, new_dist1, new_dist2 = best_move
            current_routes[r1] = new_route1
            current_routes[r2] = new_route2
            route_distances[r1] = new_dist1
            route_distances[r2] = new_dist2
            current_max = compute_max_dist(current_routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in current_routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
            improved = True
        
        # Diversification: every 50 iterations, relocate worst customer from longest route
        if it % 50 == 0 and it > 0:
            max_dist = -1
            longest_route_idx = None
            for r_idx in range(truck_count):
                d = route_distances[r_idx]
                if d > max_dist:
                    max_dist = d
                    longest_route_idx = r_idx
            if longest_route_idx is not None and len(current_routes[longest_route_idx]) > 2:
                best_gain = float('inf')
                best_pos_from = None
                for pos in range(1, len(current_routes[longest_route_idx])-1):
                    cust = current_routes[longest_route_idx][pos]
                    prev = current_routes[longest_route_idx][pos-1]
                    next_c = current_routes[longest_route_idx][pos+1]
                    gain = distance_matrix[prev, cust] + distance_matrix[cust, next_c] - distance_matrix[prev, next_c]
                    if gain < best_gain:
                        best_gain = gain
                        best_pos_from = pos
                if best_pos_from is not None:
                    cust = current_routes[longest_route_idx].pop(best_pos_from)
                    best_insert = None
                    best_insert_max = float('inf')
                    for r_to in range(truck_count):
                        if r_to == longest_route_idx:
                            continue
                        for pos_to in range(1, len(current_routes[r_to])):
                            new_dist = route_distances[r_to] + distance_matrix[current_routes[r_to][pos_to-1], cust] + distance_matrix[cust, current_routes[r_to][pos_to]] - distance_matrix[current_routes[r_to][pos_to-1], current_routes[r_to][pos_to]]
                            temp_routes = [r[:] for r in current_routes]
                            temp_routes[r_to].insert(pos_to, cust)
                            cand_max = compute_max_dist(temp_routes)
                            if cand_max < best_insert_max:
                                best_insert_max = cand_max
                                best_insert = (r_to, pos_to, new_dist)
                    if best_insert is not None:
                        r_to, pos_to, new_dist = best_insert
                        current_routes[r_to].insert(pos_to, cust)
                        route_distances[longest_route_idx] = route_distance(current_routes[longest_route_idx])
                        route_distances[r_to] = new_dist
                        current_max = compute_max_dist(current_routes)
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [r[:] for r in current_routes]
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                        improved = True
        
        if not improved:
            no_improve += 1
            deviation *= 0.98
        else:
            no_improve = 0
        
        # Restart after 100 iterations without improvement
        if no_improve >= 100:
            routes2, dists2 = construct_routes(tie_break_random=True)
            current_routes = [r[:] for r in routes2]
            route_distances = dists2[:]
            current_max = compute_max_dist(current_routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in current_routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
            deviation = 0.1 * current_max
            no_improve = 0
    
    return best_routes