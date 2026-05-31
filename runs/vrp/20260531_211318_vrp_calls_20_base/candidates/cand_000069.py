import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    dist = distance_matrix

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d

    # Initial construction: random order greedy insertion minimizing max distance
    customers = list(range(1, n))
    random.shuffle(customers)
    routes = [[0, 0] for _ in range(truck_count)]
    for cust in customers:
        best_truck = None
        best_pos = None
        best_max = float('inf')
        best_total = float('inf')
        for t, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_routes = routes[:t] + [new_route] + routes[t+1:]
                new_max = max(route_distance(r) for r in new_routes)
                new_total = sum(route_distance(r) for r in new_routes)
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max = new_max
                    best_total = new_total
                    best_truck = t
                    best_pos = pos
        routes[best_truck].insert(best_pos, cust)

    # Best solution tracking
    best_routes = [list(r) for r in routes]
    best_max = max(route_distance(r) for r in best_routes)
    best_total = sum(route_distance(r) for r in best_routes)
    report_best_vrp(best_routes)

    # Tabu search parameters
    max_iter = 1000
    tabu_tenure = int(math.sqrt(n)) + 1
    no_improve_limit = 100

    # Tabu structures: dictionary mapping customer -> remaining tenure
    tabu = {}

    current_routes = [list(r) for r in routes]
    current_max = max(route_distance(r) for r in current_routes)
    current_total = sum(route_distance(r) for r in current_routes)

    for it in range(max_iter):
        # Generate all moves
        best_move = None
        best_move_max = float('inf')
        best_move_total = float('inf')

        # Relocate moves: move one customer to a different position (same or different route)
        for t1, route1 in enumerate(current_routes):
            if len(route1) <= 2:
                continue
            for idx1 in range(1, len(route1)-1):
                cust = route1[idx1]
                # Remove cust from route1
                new_route1 = route1[:idx1] + route1[idx1+1:]
                # Consider inserting into all positions of all routes
                for t2, route2 in enumerate(current_routes):
                    for pos2 in range(1, len(route2)):
                        # Skip if same route and same position after removal
                        if t1 == t2 and pos2 == idx1:
                            continue
                        new_route2 = route2[:pos2] + [cust] + route2[pos2:]
                        new_routes = current_routes[:]
                        # Replace routes
                        if t1 == t2:
                            # Special case: after removal, route length decreased by 1, so adjust
                            # Re-inserting into same route after removal: careful
                            # Simpler: rebuild routes list
                            pass
                        # Rebuild all routes
                        new_routes_list = []
                        for t in range(truck_count):
                            if t == t1:
                                new_routes_list.append(new_route1)
                            else:
                                new_routes_list.append(list(current_routes[t]))
                        # Insert into route2 (which may be same as t1)
                        route2_new = new_routes_list[t2][:pos2] + [cust] + new_routes_list[t2][pos2:]
                        new_routes_list[t2] = route2_new

                        # Compute distances
                        new_max = max(route_distance(r) for r in new_routes_list)
                        new_total = sum(route_distance(r) for r in new_routes_list)

                        # Tabu check
                        is_tabu = cust in tabu and tabu[cust] > 0
                        # Aspiration: if yields new global best, accept
                        if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                            aspiration = True
                        else:
                            aspiration = False

                        if aspiration or not is_tabu:
                            if new_max < best_move_max or (new_max == best_move_max and new_total < best_move_total):
                                best_move_max = new_max
                                best_move_total = new_total
                                best_move = ('relocate', t1, idx1, cust, t2, pos2, new_routes_list)

        # Swap moves: exchange two customers from different routes or same route
        for t1, route1 in enumerate(current_routes):
            if len(route1) <= 2:
                continue
            for idx1 in range(1, len(route1)-1):
                cust1 = route1[idx1]
                for t2, route2 in enumerate(current_routes):
                    if len(route2) <= 2:
                        continue
                    for idx2 in range(1, len(route2)-1):
                        # Avoid duplicate pairs (order) and same customer
                        if t1 == t2 and idx1 >= idx2:
                            continue
                        cust2 = route2[idx2]
                        # Build new routes after swap
                        new_routes_list = [list(r) for r in current_routes]
                        # Swap positions
                        new_routes_list[t1][idx1] = cust2
                        new_routes_list[t2][idx2] = cust1
                        new_max = max(route_distance(r) for r in new_routes_list)
                        new_total = sum(route_distance(r) for r in new_routes_list)

                        # Tabu: if either customer is tabu and not aspiration
                        is_tabu = (cust1 in tabu and tabu[cust1] > 0) or (cust2 in tabu and tabu[cust2] > 0)
                        if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                            aspiration = True
                        else:
                            aspiration = False

                        if aspiration or not is_tabu:
                            if new_max < best_move_max or (new_max == best_move_max and new_total < best_move_total):
                                best_move_max = new_max
                                best_move_total = new_total
                                best_move = ('swap', t1, idx1, cust1, t2, idx2, cust2, new_routes_list)

        # Apply best move if found
        if best_move is None:
            break

        # Update tabu tenures: decrement all
        for c in tabu:
            tabu[c] = max(0, tabu[c] - 1)

        # Apply move and update tabu
        if best_move[0] == 'relocate':
            _, t1, idx1, cust, t2, pos2, new_routes_list = best_move
            current_routes = new_routes_list
            # Tabu the moved customer
            tabu[cust] = tabu_tenure
        else:
            _, t1, idx1, cust1, t2, idx2, cust2, new_routes_list = best_move
            current_routes = new_routes_list
            tabu[cust1] = tabu_tenure
            tabu[cust2] = tabu_tenure

        current_max = best_move_max
        current_total = best_move_total

        # Update best solution
        if current_max < best_max - 1e-9 or (abs(current_max - best_max) < 1e-9 and current_total < best_total):
            best_max = current_max
            best_total = current_total
            best_routes = [list(r) for r in current_routes]
            report_best_vrp(best_routes)
            no_improve_iter = 0
        else:
            no_improve_iter += 1

        # Diversification: random restart if no improvement for a while
        if no_improve_iter >= no_improve_limit:
            # Shake: remove a random subset of customers and reinsert greedily
            all_customers = [c for r in current_routes for c in r[1:-1]]
            random.shuffle(all_customers)
            num_remove = max(1, int(0.5 * len(all_customers)))
            to_remove = set(all_customers[:num_remove])
            partial_routes = []
            for route in current_routes:
                new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
                partial_routes.append(new_route)
            # Reinsert removed customers greedily
            remaining = [c for c in all_customers if c in to_remove]
            for cust in remaining:
                best_max_local = float('inf')
                best_total_local = float('inf')
                best_truck = None
                best_pos = None
                for t, route in enumerate(partial_routes):
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_routes = partial_routes[:t] + [new_route] + partial_routes[t+1:]
                        new_max_val = max(route_distance(r) for r in new_routes)
                        new_total_val = sum(route_distance(r) for r in new_routes)
                        if new_max_val < best_max_local or (new_max_val == best_max_local and new_total_val < best_total_local):
                            best_max_local = new_max_val
                            best_total_local = new_total_val
                            best_truck = t
                            best_pos = pos
                partial_routes[best_truck].insert(best_pos, cust)
            current_routes = partial_routes
            current_max = max(route_distance(r) for r in current_routes)
            current_total = sum(route_distance(r) for r in current_routes)
            # Reset tabu
            tabu.clear()
            no_improve_iter = 0

    return best_routes