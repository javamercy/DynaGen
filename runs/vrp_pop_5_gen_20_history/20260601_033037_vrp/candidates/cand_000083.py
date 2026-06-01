import numpy as np
import random
import collections

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    best_routes = None
    best_max = float('inf')

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]

    # Construction: min-max insertion
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(customers)
    while unassigned:
        best_cust = None
        best_route_idx = None
        best_pos = None
        best_new_max = float('inf')
        sorted_cust = sorted(unassigned)
        for cust in sorted_cust:
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_route_dist = route_distance(new_route)
                    temp_routes = [list(r) for r in routes]
                    temp_routes[r_idx] = new_route
                    new_max = max(route_distance(r) for r in temp_routes)
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_cust = cust
                        best_route_idx = r_idx
                        best_pos = pos
                    elif abs(new_max - best_new_max) < 1e-12:
                        if cust < best_cust or (cust == best_cust and r_idx < best_route_idx) or \
                           (cust == best_cust and r_idx == best_route_idx and pos < best_pos):
                            best_new_max = new_max
                            best_cust = cust
                            best_route_idx = r_idx
                            best_pos = pos
        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    report_best_vrp(routes)

    # Tabu search parameters
    max_iter = min(500, n * truck_count)
    tabu_tenure = max(5, n // 10)
    tabu_list = collections.deque(maxlen=tabu_tenure)
    max_moves_per_iter = min(100, n * truck_count)

    for _ in range(max_iter):
        improved = False
        # Find longest route
        dists = [route_distance(r) for r in routes]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        route_max = routes[max_idx]
        interior = route_max[1:-1]
        if not interior:
            break

        # Generate random moves
        moves = []  # each element: (type, data, new_routes, new_max)
        evaluated = set()
        attempts = 0
        while len(moves) < max_moves_per_iter and attempts < max_moves_per_iter * 10:
            attempts += 1
            # Decide relocate or swap with 50% probability
            if random.random() < 0.5:
                # Relocate
                cust = random.choice(interior)
                other_idx = random.randrange(truck_count)
                while other_idx == max_idx:
                    other_idx = random.randrange(truck_count)
                other_route = routes[other_idx]
                pos = random.randrange(1, len(other_route))
                move_key = (cust, max_idx, other_idx, pos)
                if move_key in evaluated:
                    continue
                evaluated.add(move_key)
                # Check tabu
                tabu_key = (cust, max_idx, other_idx)
                if tabu_key in tabu_list:
                    continue
                new_routes = [list(r) for r in routes]
                new_routes[max_idx].remove(cust)
                new_routes[other_idx].insert(pos, cust)
                new_max = max(route_distance(r) for r in new_routes)
                moves.append(('relocate', cust, max_idx, other_idx, pos, new_routes, new_max))
            else:
                # Swap
                other_idx = random.randrange(truck_count)
                while other_idx == max_idx:
                    other_idx = random.randrange(truck_count)
                other_interior = routes[other_idx][1:-1]
                if not other_interior:
                    continue
                cust_max = random.choice(interior)
                cust_other = random.choice(other_interior)
                move_key = (cust_max, cust_other, max_idx, other_idx)
                if move_key in evaluated:
                    continue
                evaluated.add(move_key)
                # Check tabu
                tabu_key1 = (cust_max, cust_other)
                tabu_key2 = (cust_other, cust_max)
                if tabu_key1 in tabu_list or tabu_key2 in tabu_list:
                    continue
                new_routes = [list(r) for r in routes]
                idx_max = new_routes[max_idx].index(cust_max)
                idx_other = new_routes[other_idx].index(cust_other)
                new_routes[max_idx][idx_max] = cust_other
                new_routes[other_idx][idx_other] = cust_max
                new_max = max(route_distance(r) for r in new_routes)
                moves.append(('swap', cust_max, max_idx, cust_other, other_idx, new_routes, new_max))

        if moves:
            # Select best move (min new_max, tie-break by first element)
            moves.sort(key=lambda m: (m[-1], m[0], m[1] if m[0]=='relocate' else m[1]))
            best_move = moves[0]
            if best_move[-1] < best_max - 1e-12 or True:  # always accept since non-tabu
                if best_move[0] == 'relocate':
                    _, cust, from_idx, to_idx, pos, new_routes, _ = best_move
                    tabu_list.append((cust, from_idx, to_idx))
                else:
                    _, cust_max, max_idx, cust_other, other_idx, new_routes, _ = best_move
                    tabu_list.append((cust_max, cust_other))
                routes = new_routes
                report_best_vrp(routes)
                improved = True
        else:
            break

        # Intra-route 2-opt on each route after move
        for idx in range(truck_count):
            route = routes[idx]
            if len(route) <= 3:
                continue
            best_route = route[:]
            best_dist = route_distance(route)
            found = False
            for a in range(1, len(route)-2):
                for b in range(a+1, len(route)-1):
                    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist - 1e-12:
                        best_dist = new_dist
                        best_route = new_route
                        found = True
                        break
                if found:
                    break
            if found:
                routes[idx] = best_route
                new_max = max(route_distance(r) for r in routes)
                if new_max < best_max - 1e-12:
                    report_best_vrp(routes)

        if not improved:
            break

    return best_routes if best_routes is not None else routes