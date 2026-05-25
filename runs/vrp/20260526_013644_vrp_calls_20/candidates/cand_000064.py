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

    def build_solution(random_tie=False):
        routes = [[0, 0] for _ in range(truck_count)]
        route_distances = [0.0 for _ in range(truck_count)]
        unassigned = set(customers)
        while unassigned:
            candidates = []
            for cust in unassigned:
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
                candidates.append((regret, cust, best[1], best[2], best[0]))
            if random_tie:
                max_regret = max(c[0] for c in candidates)
                tied = [c for c in candidates if c[0] == max_regret]
                chosen = random.choice(tied)
            else:
                candidates.sort(key=lambda x: (-x[0], x[1]))
                chosen = candidates[0]
            regret, cust, r_idx, pos, new_dist = chosen
            routes[r_idx].insert(pos, cust)
            route_distances[r_idx] = new_dist
            unassigned.remove(cust)
        return routes, route_distances

    # Initial construction with deterministic tie-breaking
    routes, route_distances = build_solution(random_tie=False)
    best_routes = [r[:] for r in routes]
    best_max = compute_max_dist(routes)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    current_routes = [r[:] for r in routes]
    current_max = best_max
    deviation = 0.2 * current_max
    total_iter = 500
    stagnation_counter = 0
    restart_flag = False

    for it in range(total_iter):
        improved = False

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
                next = current_routes[r_from][pos_from+1]
                delta_from = distance_matrix[prev, next] - distance_matrix[prev, cust] - distance_matrix[cust, next]
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

        # Diversification: every 20 iterations, move multiple customers from longest to shortest route
        if it % 20 == 0 and it > 0:
            # identify longest and shortest routes by distance
            dists = [route_distances[r] for r in range(truck_count)]
            longest_idx = max(range(truck_count), key=lambda i: dists[i])
            shortest_idx = min(range(truck_count), key=lambda i: dists[i])
            longest_route = current_routes[longest_idx]
            if len(longest_route) > 2:
                # choose up to 3 customers to move (excluding depots)
                num_to_move = min(3, len(longest_route)-2)
                # select random positions (excluding ends)
                move_positions = random.sample(range(1, len(longest_route)-1), num_to_move)
                # sort them in descending order to pop without index issues
                move_positions.sort(reverse=True)
                for pos in move_positions:
                    cust = longest_route.pop(pos)
                    # insert into shortest route at best feasible position
                    shortest_route = current_routes[shortest_idx]
                    best_insert_pos = None
                    best_insert_cost = float('inf')
                    for insert_pos in range(1, len(shortest_route)):
                        prev = shortest_route[insert_pos-1]
                        next = shortest_route[insert_pos]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, next] - distance_matrix[prev, next]
                        if delta < best_insert_cost:
                            best_insert_cost = delta
                            best_insert_pos = insert_pos
                    shortest_route.insert(best_insert_pos, cust)
                    # update route distances
                    route_distances[longest_idx] = route_distance(longest_route)
                    route_distances[shortest_idx] = route_distance(shortest_route)
                    # update current max
                    current_max = compute_max_dist(current_routes)
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in current_routes]
                        try:
                            report_best_vrp(best_routes)
                        except NameError:
                            pass
                    improved = True

        if improved:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            deviation *= 0.99

        # Restart if stagnation for 60 iterations
        if stagnation_counter >= 60:
            new_routes, new_distances = build_solution(random_tie=restart_flag)
            current_routes = [r[:] for r in new_routes]
            route_distances = new_distances[:]
            current_max = compute_max_dist(current_routes)
            deviation = 0.2 * current_max
            stagnation_counter = 0
            restart_flag = not restart_flag

    return best_routes