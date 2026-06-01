import numpy as np
import math
import random

def solve_vrp(distance_matrix, truck_count):
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    # Construction: regret-2 insertion minimizing max route distance
    def route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def construct_solution():
        # Start with empty routes (only depots)
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        uninserted = set(customers)

        while uninserted:
            best_regret = -1.0
            best_cust = -1
            best_route = -1
            best_pos = -1
            # Precompute current max among routes (excluding candidate)
            current_max = max(route_dists) if route_dists else 0.0
            for cust in uninserted:
                insertion_costs = []
                for ri, route in enumerate(routes):
                    # Positions where customer can be inserted (between depots or between customers)
                    for pos in range(1, len(route)):
                        # Compute new distance for this route
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_dist = route_length(new_route)
                        # Compute new max distance across all routes
                        # Max of other routes distances and new route distance
                        other_max = 0.0
                        for rj, d in enumerate(route_dists):
                            if rj != ri and d > other_max:
                                other_max = d
                        new_max = max(new_dist, other_max)
                        insertion_costs.append((new_max, ri, pos))
                if not insertion_costs:
                    continue
                # Sort by new_max, then by route index, then by position
                insertion_costs.sort(key=lambda x: (x[0], x[1], x[2]))
                best_cost = insertion_costs[0][0]
                second_cost = insertion_costs[1][0] if len(insertion_costs) > 1 else best_cost
                regret = second_cost - best_cost
                # Use regret and tie-break by customer id (smaller first)
                if regret > best_regret or (regret == best_regret and cust < best_cust):
                    best_regret = regret
                    best_cust = cust
                    best_route = insertion_costs[0][1]
                    best_pos = insertion_costs[0][2]
            # Insert the selected customer
            routes[best_route] = routes[best_route][:best_pos] + [best_cust] + routes[best_route][best_pos:]
            route_dists[best_route] = route_length(routes[best_route])
            uninserted.remove(best_cust)
        return routes

    routes = construct_solution()

    def compute_max():
        return max(sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1)) for route in routes)

    def copy_routes():
        return [list(r) for r in routes]

    best_routes = copy_routes()
    best_max = compute_max()
    report_best_vrp(best_routes)

    # Operator definitions (modify routes in place, return True if overall max improved)
    def op_2opt():
        nonlocal routes, best_max, best_routes
        for ri, route in enumerate(routes):
            if len(route) <= 3:
                continue
            best_local_dist = route_dist(route)
            best_local_route = route[:]
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < best_local_dist - 1e-9:
                        best_local_dist = new_dist
                        best_local_route = new_route
                        improved = True
            if improved:
                routes[ri] = best_local_route
                new_max = compute_max()
                if new_max < best_max - 1e-9:
                    best_max = new_max
                    best_routes = copy_routes()
                    report_best_vrp(best_routes)
                return True
        return False

    def op_relocate():
        nonlocal routes, best_max, best_routes
        dists = [route_dist(r) for r in routes]
        longest_idx = max(range(len(routes)), key=lambda i: dists[i])
        src_route = routes[longest_idx]
        if len(src_route) <= 2:
            return False
        for pos_i in range(1, len(src_route) - 1):
            cust = src_route[pos_i]
            for dst_idx in range(len(routes)):
                if dst_idx == longest_idx:
                    continue
                dst_route = routes[dst_idx]
                for pos_j in range(1, len(dst_route)):
                    new_src = src_route[:pos_i] + src_route[pos_i+1:]
                    new_dst = dst_route[:pos_j] + [cust] + dst_route[pos_j:]
                    new_dists = [route_dist(r) for ri, r in enumerate(routes) if ri not in (longest_idx, dst_idx)]
                    new_dist_src = route_dist(new_src)
                    new_dist_dst = route_dist(new_dst)
                    new_max = max([new_dist_src, new_dist_dst] + new_dists)
                    if new_max < compute_max() - 1e-9:
                        routes[longest_idx] = new_src
                        routes[dst_idx] = new_dst
                        if new_max < best_max - 1e-9:
                            best_max = new_max
                            best_routes = copy_routes()
                            report_best_vrp(best_routes)
                        return True
        return False

    def op_swap():
        nonlocal routes, best_max, best_routes
        dists = [route_dist(r) for r in routes]
        longest_idx = max(range(len(routes)), key=lambda i: dists[i])
        src_route = routes[longest_idx]
        if len(src_route) <= 2:
            return False
        for pos_i in range(1, len(src_route) - 1):
            cust_i = src_route[pos_i]
            for dst_idx in range(len(routes)):
                if dst_idx == longest_idx:
                    continue
                dst_route = routes[dst_idx]
                if len(dst_route) <= 2:
                    continue
                for pos_j in range(1, len(dst_route) - 1):
                    cust_j = dst_route[pos_j]
                    new_src = src_route[:pos_i] + [cust_j] + src_route[pos_i+1:]
                    new_dst = dst_route[:pos_j] + [cust_i] + dst_route[pos_j+1:]
                    new_dists = [route_dist(r) for ri, r in enumerate(routes) if ri not in (longest_idx, dst_idx)]
                    new_dist_src = route_dist(new_src)
                    new_dist_dst = route_dist(new_dst)
                    new_max = max([new_dist_src, new_dist_dst] + new_dists)
                    if new_max < compute_max() - 1e-9:
                        routes[longest_idx] = new_src
                        routes[dst_idx] = new_dst
                        if new_max < best_max - 1e-9:
                            best_max = new_max
                            best_routes = copy_routes()
                            report_best_vrp(best_routes)
                        return True
        return False

    operators = [op_2opt, op_relocate, op_swap]
    num_ops = len(operators)
    scores = [1.0] * num_ops
    max_iter = 200 * n
    no_improve_limit = 5 * n
    no_improve_count = 0
    ruin_count = 0
    max_ruin = 5

    def route_dist(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def compute_max_with_route(routes, ri, new_route):
        max_dist = 0.0
        for i, r in enumerate(routes):
            d = route_dist(new_route) if i == ri else route_dist(r)
            if d > max_dist:
                max_dist = d
        return max_dist

    def ruin_and_recreate():
        nonlocal routes, best_max, best_routes, no_improve_count, scores, ruin_count
        m_customers = list(range(1, n))
        if len(m_customers) < 3:
            return False
        num_remove = max(1, int(0.3 * len(m_customers)))
        customers_removed = random.sample(m_customers, num_remove)
        # Remove these customers from routes
        for route in routes:
            for cust in customers_removed:
                if cust in route:
                    route.remove(cust)
        # Reinsert each removed customer greedily minimizing new max
        for cust in customers_removed:
            best_new_max = math.inf
            best_ri = -1
            best_pos = -1
            for ri, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_max = compute_max_with_route(routes, ri, new_route)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_ri = ri
                        best_pos = pos
            if best_ri != -1:
                routes[best_ri] = routes[best_ri][:best_pos] + [cust] + routes[best_ri][best_pos:]
            else:
                routes[0].insert(len(routes[0])-1, cust)
        new_max = compute_max()
        if new_max < best_max - 1e-9:
            best_max = new_max
            best_routes = copy_routes()
            report_best_vrp(best_routes)
        no_improve_count = 0
        scores = [1.0] * num_ops
        ruin_count += 1
        return True

    for iteration in range(max_iter):
        if no_improve_count >= no_improve_limit and ruin_count < max_ruin:
            ruin_and_recreate()
            continue
        elif no_improve_count >= no_improve_limit:
            break
        # Adaptive selection
        total_score = sum(scores)
        r = random.random() * total_score
        cumulative = 0.0
        op_idx = 0
        for idx, score in enumerate(scores):
            cumulative += score
            if r <= cumulative:
                op_idx = idx
                break
        # Apply operator
        improved = operators[op_idx]()
        if improved:
            scores[op_idx] *= 1.1
            no_improve_count = 0
        else:
            scores[op_idx] *= 0.9
            no_improve_count += 1

    return best_routes