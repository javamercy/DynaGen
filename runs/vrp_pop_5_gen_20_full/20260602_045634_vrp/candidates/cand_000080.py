import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)
    
    def repair(routes):
        for r in routes:
            if r[0] != 0:
                r.insert(0, 0)
            if r[-1] != 0:
                r.append(0)
        return routes
    
    def construct():
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in customers:
            best_increase = float('inf')
            best_moves = []
            current_max = max_route_distance(routes)
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_route_dist = route_distance(route) + added
                    other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                    new_max = max(new_route_dist, other_max)
                    increase = new_max - current_max
                    if increase < best_increase - 1e-12:
                        best_increase = increase
                        best_moves = [(r_idx, pos)]
                    elif abs(increase - best_increase) < 1e-12:
                        best_moves.append((r_idx, pos))
            best_route, best_pos = random.choice(best_moves)
            routes[best_route].insert(best_pos, cust)
        return routes
    
    def vnd(routes):
        best_routes = [r[:] for r in routes]
        best_max = max_route_distance(routes)
        improved = True
        max_iter = (n - 1) * truck_count * 2
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            # Run all 5 phases in random order
            phases = ['2opt', 'relocate', 'swap', 'cross', '2optstar']
            random.shuffle(phases)
            for phase in phases:
                if phase == '2opt':
                    for r_idx in range(truck_count):
                        route = routes[r_idx]
                        if len(route) <= 3:
                            continue
                        for i in range(1, len(route)-2):
                            for j in range(i+1, len(route)-1):
                                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                                old_dist = route_distance(route)
                                new_dist = route_distance(new_route)
                                if new_dist >= old_dist:
                                    continue
                                other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != r_idx)
                                new_max = max(new_dist, other_max)
                                if new_max < best_max - 1e-12:
                                    routes[r_idx] = new_route
                                    best_routes = [r[:] for r in routes]
                                    best_max = new_max
                                    improved = True
                                    report_best_vrp(best_routes)
                                    break
                            if improved:
                                break
                        if improved:
                            break
                elif phase == 'relocate':
                    for src in range(truck_count):
                        route_src = routes[src]
                        if len(route_src) <= 2:
                            continue
                        for pos_src in range(1, len(route_src)-1):
                            cust = route_src[pos_src]
                            temp_src = route_src[:pos_src] + route_src[pos_src+1:]
                            dist_src = route_distance(temp_src)
                            for dst in range(truck_count):
                                if dst == src:
                                    continue
                                route_dst = routes[dst]
                                for pos_dst in range(1, len(route_dst)):
                                    new_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                                    dist_dst = route_distance(new_dst)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != src and x != dst)
                                    new_max = max(dist_src, dist_dst, other_max)
                                    if new_max < best_max - 1e-12:
                                        routes[src] = temp_src
                                        routes[dst] = new_dst
                                        best_routes = [r[:] for r in routes]
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                elif phase == 'swap':
                    for t1 in range(truck_count):
                        route1 = routes[t1]
                        if len(route1) <= 2:
                            continue
                        for t2 in range(t1+1, truck_count):
                            route2 = routes[t2]
                            if len(route2) <= 2:
                                continue
                            for i in range(1, len(route1)-1):
                                for j in range(1, len(route2)-1):
                                    cust1 = route1[i]
                                    cust2 = route2[j]
                                    new_route1 = route1[:i] + [cust2] + route1[i+1:]
                                    new_route2 = route2[:j] + [cust1] + route2[j+1:]
                                    dist1 = route_distance(new_route1)
                                    dist2 = route_distance(new_route2)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                    new_max = max(dist1, dist2, other_max)
                                    if new_max < best_max - 1e-12:
                                        routes[t1] = new_route1
                                        routes[t2] = new_route2
                                        best_routes = [r[:] for r in routes]
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                elif phase == 'cross':
                    for t1 in range(truck_count):
                        route1 = routes[t1]
                        if len(route1) <= 2:
                            continue
                        for t2 in range(t1+1, truck_count):
                            route2 = routes[t2]
                            if len(route2) <= 2:
                                continue
                            for i in range(1, len(route1)-1):
                                for j in range(1, len(route2)-1):
                                    new_route1 = route1[:i] + route2[j:]
                                    new_route2 = route2[:j] + route1[i:]
                                    # ensure depot at both ends
                                    if new_route1[0] != 0:
                                        new_route1.insert(0, 0)
                                    if new_route1[-1] != 0:
                                        new_route1.append(0)
                                    if new_route2[0] != 0:
                                        new_route2.insert(0, 0)
                                    if new_route2[-1] != 0:
                                        new_route2.append(0)
                                    dist1 = route_distance(new_route1)
                                    dist2 = route_distance(new_route2)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                    new_max = max(dist1, dist2, other_max)
                                    if new_max < best_max - 1e-12:
                                        routes[t1] = new_route1
                                        routes[t2] = new_route2
                                        best_routes = [r[:] for r in routes]
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                elif phase == '2optstar':
                    for t1 in range(truck_count):
                        route1 = routes[t1]
                        if len(route1) <= 2:
                            continue
                        for t2 in range(t1+1, truck_count):
                            route2 = routes[t2]
                            if len(route2) <= 2:
                                continue
                            for i in range(1, len(route1)-1):
                                for j in range(1, len(route2)-1):
                                    new_route1 = route1[:i] + route2[j:]
                                    new_route2 = route2[:j] + route1[i:]
                                    if new_route1[0] != 0:
                                        new_route1.insert(0, 0)
                                    if new_route1[-1] != 0:
                                        new_route1.append(0)
                                    if new_route2[0] != 0:
                                        new_route2.insert(0, 0)
                                    if new_route2[-1] != 0:
                                        new_route2.append(0)
                                    dist1 = route_distance(new_route1)
                                    dist2 = route_distance(new_route2)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                    new_max = max(dist1, dist2, other_max)
                                    if new_max < best_max - 1e-12:
                                        routes[t1] = new_route1
                                        routes[t2] = new_route2
                                        best_routes = [r[:] for r in routes]
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                if improved:
                    break
        return best_routes
    
    def perturb_block(routes, intensity):
        for _ in range(intensity):
            src = random.randint(0, truck_count-1)
            while len(routes[src]) <= 3:
                src = random.randint(0, truck_count-1)
            block_len = random.randint(1, min(3, len(routes[src])-2))
            start = random.randint(1, len(routes[src])-1-block_len)
            block = routes[src][start:start+block_len]
            del routes[src][start:start+block_len]
            dst = random.randint(0, truck_count-1)
            pos = random.randint(1, len(routes[dst])-1)
            for c in block:
                routes[dst].insert(pos, c)
                pos += 1
        return routes
    
    def perturb_ruin_recreate(routes, intensity):
        # Remove a random subset of customers (size between 5 and 15 or 10% of n, whichever larger)
        num_customers = n - 1
        remove_count = random.randint(min(5, num_customers), min(15, num_customers // 2))
        # gather all customers
        all_customers = []
        for r in routes:
            for c in r:
                if c != 0:
                    all_customers.append(c)
        random.shuffle(all_customers)
        to_remove = set(all_customers[:remove_count])
        # remove them from routes
        for r in routes:
            for c in list(to_remove):
                if c in r:
                    r.remove(c)
        # repair routes to have 0 at ends
        for r in routes:
            if r[0] != 0:
                r.insert(0, 0)
            if r[-1] != 0:
                r.append(0)
        # reinsert removed customers using best insertion with random tie-breaking
        for cust in to_remove:
            best_increase = float('inf')
            best_moves = []
            current_max = max_route_distance(routes)
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_route_dist = route_distance(route) + added
                    other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                    new_max = max(new_route_dist, other_max)
                    increase = new_max - current_max
                    if increase < best_increase - 1e-12:
                        best_increase = increase
                        best_moves = [(r_idx, pos)]
                    elif abs(increase - best_increase) < 1e-12:
                        best_moves.append((r_idx, pos))
            best_route, best_pos = random.choice(best_moves)
            routes[best_route].insert(best_pos, cust)
        return routes
    
    global_best_routes = None
    global_best_max = float('inf')
    restarts = max(10, n // 5)  # more restarts for diversity
    temperature = 100.0
    temp_decay = 0.95
    for restart in range(restarts):
        # construction
        routes = construct()
        routes = vnd(routes)
        current_max = max_route_distance(routes)
        # accept if better or according to SA
        if current_max < global_best_max - 1e-12:
            global_best_max = current_max
            global_best_routes = [r[:] for r in routes]
        else:
            delta = current_max - global_best_max
            if delta > 0 and random.random() < math.exp(-delta / temperature):
                # accept worse as new start
                pass  # keep current routes as starting point for perturbation
            else:
                # revert to global best for perturbation
                if global_best_routes is not None:
                    routes = [r[:] for r in global_best_routes]
        # perturbation for next iteration
        if restart < restarts - 1:
            intensity = 1 + (restart // 5)  # increase intensity over time
            # randomly choose perturbation type
            if random.random() < 0.5:
                routes = perturb_block(routes, intensity)
            else:
                routes = perturb_ruin_recreate(routes, intensity)
        temperature *= temp_decay
    
    if global_best_routes is None:
        global_best_routes = [[0, 0] for _ in range(truck_count)]
    return repair(global_best_routes)