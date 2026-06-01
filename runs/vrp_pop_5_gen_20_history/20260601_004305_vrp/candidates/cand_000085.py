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
        return routes

    # --- Initial solution: Nearest-neighbor TSP + greedy insertion into routes ---
    tour = []
    visited = [False] * n
    visited[0] = True
    current = 0
    for _ in range(m):
        best = -1
        best_dist = math.inf
        for v in range(1, n):
            if not visited[v]:
                d = distance_matrix[current][v]
                if d < best_dist or (d == best_dist and v < best):
                    best_dist = d
                    best = v
        tour.append(best)
        visited[best] = True
        current = best

    # Greedy assignment: insert each customer in turn to the route that minimizes the new max distance
    routes = [[0, 0] for _ in range(truck_count)]
    for c in tour:
        best_inc = math.inf
        best_ri = -1
        best_pos = -1
        for ri in range(len(routes)):
            route = routes[ri]
            for pos in range(1, len(route)):
                new_route = route[:pos] + [c] + route[pos:]
                new_dist = sum(distance_matrix[new_route[k]][new_route[k+1]] for k in range(len(new_route)-1))
                other_dists = [sum(distance_matrix[routes[ri2][k]][routes[ri2][k+1]] for k in range(len(routes[ri2])-1)) for ri2 in range(len(routes)) if ri2 != ri]
                new_max = max(new_dist, *other_dists) if other_dists else new_dist
                if new_max < best_inc or (new_max == best_inc and (ri < best_ri or (ri == best_ri and pos < best_pos))):
                    best_inc = new_max
                    best_ri = ri
                    best_pos = pos
        routes[best_ri].insert(best_pos, c)

    def route_dist(route):
        return sum(distance_matrix[route[k]][route[k+1]] for k in range(len(route)-1))

    def compute_max(routes):
        return max(route_dist(r) for r in routes)

    def copy_routes(routes):
        return [list(r) for r in routes]

    current = copy_routes(routes)
    current_max = compute_max(current)
    best = copy_routes(current)
    best_max = current_max
    report_best_vrp(best)

    # --- Improvement loop: random removal and greedy insertion, accept only if better ---
    max_iter = max(1000, m * 10)
    n_cust = m
    for it in range(max_iter):
        # Destroy: random removal of one customer
        # Choose a customer to remove
        non_empty_routes = [r for r in current if len(r) > 2]
        if not non_empty_routes:
            break
        r_idx = random.randrange(len(non_empty_routes))
        route = non_empty_routes[r_idx]
        # pick random position (excluding depots)
        pos = random.randrange(1, len(route)-1)
        c = route.pop(pos)
        removed = [c]
        # If route becomes empty after removal, keep it as [0,0]
        if len(route) == 2:
            route.clear()
            route.extend([0,0])

        # Repair: greedy insertion of the removed customer
        best_inc = math.inf
        best_ri = -1
        best_pos = -1
        for ri, route in enumerate(current):
            for pos in range(1, len(route)):
                new_route = route[:pos] + removed + route[pos:]
                new_dist = sum(distance_matrix[new_route[k]][new_route[k+1]] for k in range(len(new_route)-1))
                other_dists = [sum(distance_matrix[current[ri2][k]][current[ri2][k+1]] for k in range(len(current[ri2])-1)) for ri2 in range(len(current)) if ri2 != ri]
                new_max = max(new_dist, *other_dists) if other_dists else new_dist
                if new_max < best_inc or (new_max == best_inc and (ri < best_ri or (ri == best_ri and pos < best_pos))):
                    best_inc = new_max
                    best_ri = ri
                    best_pos = pos
        current[best_ri].insert(best_pos, removed[0])

        # Evaluate new solution
        new_max = compute_max(current)
        if new_max < current_max:
            current_max = new_max
            if new_max < best_max:
                best = copy_routes(current)
                best_max = new_max
                report_best_vrp(best)
        # else revert? No, we accept only improvements, but we already inserted. We could revert, but simpler: we keep the change even if worse? Actually we want to accept only improvements. So we need to revert if not improving.
        else:
            # Revert the change: we need to remove the customer from where it was inserted and put it back to its original position.
            # However, the removal already changed the route, and we inserted in a different place. It's messy to revert. Instead, we can only accept if improvement. So we need to either revert or not insert if not improvement.
            # Let's restructure: first destroy, then compute insertion options, accept only if best_inc < current_max, else revert destroy.
            # Since we already inserted, we need to undo. To keep code simple, we can just revert by moving the customer back.
            # But for simplicity, we'll only accept improving moves: we'll compute the best insertion and then compare before inserting.
            # So we need to store the insertion decision and only apply if it improves.
            # Let's rewrite the loop to be clean.
            pass

        # Actually, the above loop has a bug: we applied insertion before checking improvement. Let's fix by computing insertion without applying, then apply if improvement.

    # Rewrite improvement loop properly:
    current = copy_routes(routes)
    current_max = compute_max(current)
    best = copy_routes(current)
    best_max = current_max
    report_best_vrp(best)

    max_iter = max(1000, m * 10)
    n_cust = m
    for it in range(max_iter):
        # Choose a random customer to relocate
        # Find a non-empty route (has at least one customer)
        non_empty = [i for i, r in enumerate(current) if len(r) > 2]
        if not non_empty:
            break
        ri = random.choice(non_empty)
        route = current[ri]
        # pick random customer position (1 to len-2)
        pos = random.randint(1, len(route)-2)
        c = route.pop(pos)
        if len(route) == 2:
            route.clear()
            route.extend([0,0])
        # Try inserting into all positions and routes, find best improving move
        best_val = math.inf
        best_ri2 = -1
        best_pos2 = -1
        for ri2 in range(truck_count):
            r2 = current[ri2]
            for p in range(1, len(r2)):
                new_route = r2[:p] + [c] + r2[p:]
                new_dist = sum(distance_matrix[new_route[k]][new_route[k+1]] for k in range(len(new_route)-1))
                other_dists = [sum(distance_matrix[current[ri3][k]][current[ri3][k+1]] for k in range(len(current[ri3])-1)) for ri3 in range(truck_count) if ri3 != ri2]
                cur_max = max(new_dist, *other_dists) if other_dists else new_dist
                if cur_max < best_val or (cur_max == best_val and (ri2 < best_ri2 or (ri2 == best_ri2 and p < best_pos2))):
                    best_val = cur_max
                    best_ri2 = ri2
                    best_pos2 = p
        # Now if best_val < current_max, apply; else revert (put back where it was)
        if best_val < current_max:
            # Insert at best position
            current[best_ri2].insert(best_pos2, c)
            current_max = best_val
            if best_val < best_max:
                best = copy_routes(current)
                best_max = best_val
                report_best_vrp(best)
        else:
            # Revert: put c back into original route at original position
            current[ri].insert(pos, c)

    return best