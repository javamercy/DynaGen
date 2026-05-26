import numpy as np
from copy import deepcopy

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))

    def route_dist(route):
        dist = 0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist

    def best_insert(route, cust):
        best_increase = float('inf')
        best_pos = -1
        for pos in range(1, len(route)):
            increase = (distance_matrix[route[pos-1], cust] +
                        distance_matrix[cust, route[pos]] -
                        distance_matrix[route[pos-1], route[pos]])
            if increase < best_increase:
                best_increase = increase
                best_pos = pos
        return best_pos, best_increase

    # Construction phase
    while unassigned:
        regrets = []
        for cust in unassigned:
            increases = []
            for ridx, route in enumerate(routes):
                _, inc = best_insert(route, cust)
                increases.append(inc)
            increases.sort()
            if len(increases) >= 2:
                regret = increases[1] - increases[0]
            else:
                regret = 1e9  # large regret if only one route
            best_inc = increases[0] if increases else float('inf')
            regrets.append((regret, best_inc, cust))
        # Choose customer with max regret, tie-break on larger best_inc, then larger cust
        regrets.sort(key=lambda x: (-x[0], -x[1], -x[2]))
        chosen_cust = regrets[0][2]
        # Insert at best position in best route
        best_route_idx = -1
        best_pos = -1
        best_inc = float('inf')
        for ridx, route in enumerate(routes):
            pos, inc = best_insert(route, chosen_cust)
            if inc < best_inc:
                best_inc = inc
                best_pos = pos
                best_route_idx = ridx
        routes[best_route_idx].insert(best_pos, chosen_cust)
        unassigned.remove(chosen_cust)

    # Evaluate initial solution
    dists = [route_dist(r) for r in routes]
    best_makespan = max(dists)
    best_routes = deepcopy(routes)
    report_best_vrp(best_routes)

    # Improvement: relocate moves
    n_customers = n - 1
    max_moves = 2 * n_customers
    for _ in range(max_moves):
        improved = False
        for cust in range(1, n):
            # Find current route of cust
            curr_route_idx = None
            for ridx, route in enumerate(routes):
                if cust in route:
                    curr_route_idx = ridx
                    break
            if curr_route_idx is None:
                continue
            curr_route = routes[curr_route_idx]
            # Remove cust from current route
            new_curr = [x for x in curr_route if x != cust]
            # Ensure route still starts and ends with 0
            if len(new_curr) < 2:
                new_curr = [0, 0]
            else:
                if new_curr[0] != 0:
                    new_curr.insert(0, 0)
                if new_curr[-1] != 0:
                    new_curr.append(0)
            old_route_dist = route_dist(curr_route)
            new_curr_dist = route_dist(new_curr)
            # Consider inserting cust into other routes (including possibly same route? But that would be a relocate within same route, which we can ignore for now)
            best_other_route_idx = -1
            best_other_pos = -1
            best_other_increase = float('inf')
            for ridx, route in enumerate(routes):
                if ridx == curr_route_idx:
                    continue
                pos, inc = best_insert(route, cust)
                if inc < best_other_increase:
                    best_other_increase = inc
                    best_other_pos = pos
                    best_other_route_idx = ridx
            if best_other_route_idx == -1:
                continue
            # Compute new distances
            new_other = routes[best_other_route_idx].copy()
            new_other.insert(best_other_pos, cust)
            new_other_dist = route_dist(new_other)
            new_makespan = max(new_curr_dist, new_other_dist,
                               max([route_dist(r) for idx, r in enumerate(routes) if idx not in (curr_route_idx, best_other_route_idx)]))
            if new_makespan < best_makespan:
                # Apply move
                routes[curr_route_idx] = new_curr
                routes[best_other_route_idx] = new_other
                best_makespan = new_makespan
                best_routes = deepcopy(routes)
                report_best_vrp(best_routes)
                improved = True
                break  # restart after first improvement
        if not improved:
            break

    return best_routes