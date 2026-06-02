import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # sort customers descending by distance from depot
    customers.sort(key=lambda c: -distance_matrix[0][c])

    # initialize routes: each [0,0]
    routes = [[0, 0] for _ in range(truck_count)]
    route_distances = [0.0] * truck_count

    # compute distance of a route
    def route_dist(route):
        if not route:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d

    # insertion: assign each customer to the route with smallest current total distance, at cheapest insertion position
    for cust in customers:
        best_route = -1
        best_pos = -1
        best_increase = float('inf')
        best_new_dist = float('inf')
        for r in range(truck_count):
            route = routes[r]
            cur_dist = route_distances[r]
            # evaluate all insertion positions (inside the route, between depots)
            for pos in range(1, len(route)):
                prev = route[pos-1]
                nxt = route[pos]
                increase = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                new_dist = cur_dist + increase
                # choose route with smallest new distance; tie: smallest route index
                if new_dist < best_new_dist or (new_dist == best_new_dist and r < best_route):
                    best_new_dist = new_dist
                    best_route = r
                    best_pos = pos
                    best_increase = increase
        # insert
        routes[best_route].insert(best_pos, cust)
        route_distances[best_route] = best_new_dist  # update immediately

    # compute initial best
    best_routes = [list(r) for r in routes]
    best_max = max(route_distances)
    # report initial best
    report_best_vrp(best_routes)

    # helper to compute distance after swapping two customers (assuming they belong to routes r1 and r2, indices i and j)
    # not needed directly, we'll recompute routes after each move

    # local search
    n_customers = len(customers)
    max_iter = n_customers * 5  # finite bound
    for iteration in range(max_iter):
        improved = False
        # --- relocate 1: move a customer from one route to another (or same route different position)
        for r1 in range(truck_count):
            route1 = routes[r1]
            if len(route1) <= 2:
                continue
            for i_idx in range(1, len(route1)-1):
                cust = route1[i_idx]
                # current neighbors
                prev_i = route1[i_idx-1]
                next_i = route1[i_idx+1]
                # removal cost
                removal_inc = distance_matrix[prev_i][next_i] - distance_matrix[prev_i][cust] - distance_matrix[cust][next_i]
                # try all target routes and positions
                for r2 in range(truck_count):
                    route2 = routes[r2]
                    # positions: from 1 to len(route2)-1 (cannot insert at start or end depots? Actually we can insert between any two nodes including depots; but inserting at position 1 means after first depot, position len(route)-1 means before last depot)
                    for pos in range(1, len(route2)):
                        if r2 == r1 and pos == i_idx:
                            continue  # same position
                        # compute new distances for affected routes
                        new_dist1 = route_distances[r1] + removal_inc
                        # for route2, insertion cost
                        prev_j = route2[pos-1]
                        next_j = route2[pos]
                        insertion_inc = distance_matrix[prev_j][cust] + distance_matrix[cust][next_j] - distance_matrix[prev_j][next_j]
                        new_dist2 = route_distances[r2] + insertion_inc
                        # new max
                        other_dists = [route_distances[r] for r in range(truck_count) if r != r1 and r != r2]
                        new_max = max(other_dists + [new_dist1, new_dist2])
                        if new_max < best_max - 1e-9:  # improvement
                            # apply move
                            # remove from route1
                            removed = routes[r1].pop(i_idx)
                            # adjust indices because removal shifts; correct: we already have index i_idx; after removal, route1 length decreases
                            # but we need to adjust for changes? We'll reconstruct: we perform removal and insertion
                            # easier: create new routes lists
                            updated_routes = [list(r) for r in routes]
                            # remove from r1 at i_idx
                            del updated_routes[r1][i_idx]
                            # insert into r2 at pos (note: if r2 == r1 and pos > i_idx, pos might shift; but we already treat removal and insertion separately; we'll do removal first then insertion on updated list)
                            # But careful: if r2 == r1, removal changes indices. Since we are doing atomic removal then insertion, we need to adjust insertion index if r2 == r1.
                            # For simplicity, skip relocate to same route? Actually we can allow relocate to same route as a different position, but the logic is more complex. We'll skip same route relocate (only cross-route) to keep it simple.
                            if r2 == r1:
                                continue
                            # insert into r2 at position pos
                            updated_routes[r2].insert(pos, cust)
                            # recompute distances
                            d1 = route_dist(updated_routes[r1])
                            d2 = route_dist(updated_routes[r2])
                            new_max_calc = max([d1, d2] + [route_dist(updated_routes[r]) for r in range(truck_count) if r != r1 and r != r2])
                            if new_max_calc < best_max - 1e-9:
                                # accept
                                routes = updated_routes
                                route_distances[r1] = d1
                                route_distances[r2] = d2
                                best_max = new_max_calc
                                best_routes = [list(r) for r in routes]
                                report_best_vrp(best_routes)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue  # restart loops

        # --- swap: exchange two customers from possibly different routes (or same route)
        for r1 in range(truck_count):
            route1 = routes[r1]
            if len(route1) <= 2:
                continue
            for i_idx in range(1, len(route1)-1):
                cust1 = route1[i_idx]
                for r2 in range(r1, truck_count):  # to avoid duplicate pairs
                    route2 = routes[r2]
                    if len(route2) <= 2:
                        continue
                    # start j_idx from appropriate index
                    start_j = 1 if r2 != r1 else i_idx+1  # ensure distinct customers in same route
                    for j_idx in range(start_j, len(route2)-1):
                        cust2 = route2[j_idx]
                        # compute new distances if swapped
                        # we need to compute the effect on route lengths
                        # For route1: swap cust1 and cust2 (but cust2 is in route2, so if r1 != r2, we need to compute insertion of cust2 into route1 and removal of cust1, and vice versa)
                        # For simplicity, we'll do a full recomputation after attempting swap on a copy
                        updated_routes = [list(r) for r in routes]
                        # swap the two customers
                        # remove from their positions first (careful if same route)
                        if r1 == r2:
                            # swap indices i_idx and j_idx in same route
                            updated_routes[r1][i_idx], updated_routes[r1][j_idx] = updated_routes[r1][j_idx], updated_routes[r1][i_idx]
                            new_dist = route_dist(updated_routes[r1])
                            other_dists = [route_dist(updated_routes[r]) for r in range(truck_count) if r != r1]
                            new_max_calc = max(other_dists + [new_dist])
                        else:
                            # cross-route swap
                            # remove first
                            del updated_routes[r1][i_idx]
                            # now indices shift; we need to adjust j_idx if r2 > r1? Actually we haven't removed from r2 yet.
                            # better: copy and swap elements
                            updated_routes[r1][i_idx] = cust2
                            updated_routes[r2][j_idx] = cust1
                            new_dist1 = route_dist(updated_routes[r1])
                            new_dist2 = route_dist(updated_routes[r2])
                            other_dists = [route_dist(updated_routes[r]) for r in range(truck_count) if r != r1 and r != r2]
                            new_max_calc = max(other_dists + [new_dist1, new_dist2])
                        if new_max_calc < best_max - 1e-9:
                            # accept
                            routes = updated_routes
                            route_distances = [route_dist(r) for r in routes]
                            best_max = new_max_calc
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # --- intra-route 2-opt
        for r in range(truck_count):
            route = routes[r]
            if len(route) <= 3:
                continue
            # try all segments [i, j] where 0 < i < j < len(route)-1 (depots excluded)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # reverse segment i to j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < route_distances[r] - 1e-9:
                        other_dists = [route_distances[rr] for rr in range(truck_count) if rr != r]
                        new_max = max(other_dists + [new_dist])
                        if new_max < best_max - 1e-9:
                            routes[r] = new_route
                            route_distances[r] = new_dist
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break  # no improvement found, exit

    return best_routes