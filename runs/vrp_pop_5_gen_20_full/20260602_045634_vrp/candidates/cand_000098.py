import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    best_global_routes = None
    best_global_max = float('inf')
    restarts = max(15, n // 3)
    
    for restart in range(restarts):
        # Construction: regret insertion with deterministic tie-breaking
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        
        for cust in customers:
            best_increase = float('inf')
            best_route = -1
            best_pos = -1
            current_max = max_distance(routes)
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
                    if increase < best_increase - 1e-12 or (abs(increase - best_increase) < 1e-12 and (r_idx < best_route or (r_idx == best_route and pos < best_pos))):
                        best_increase = increase
                        best_route = r_idx
                        best_pos = pos
            routes[best_route].insert(best_pos, cust)
        
        best_routes = [r[:] for r in routes]
        best_max = max_distance(routes)
        report_best_vrp(best_routes)
        
        # Local search cycles
        for cycle in range(5):
            max_iter = (n - 1) * truck_count * 10
            no_improve_count = 0
            for iteration in range(max_iter):
                improved = False
                max_dist = max_distance(routes)
                longest_routes = [i for i, r in enumerate(routes) if abs(route_distance(r) - max_dist) < 1e-12]
                
                # Phases: intra2opt, oropt (block moves), inter2opt, relocate, swap, cross
                for phase in ['intra2opt', 'oropt1', 'oropt2', 'oropt3', 'inter2opt', 'relocate', 'swap', 'cross']:
                    if improved:
                        break
                    if phase == 'intra2opt':
                        for r_idx in longest_routes:
                            route = routes[r_idx]
                            if len(route) <= 3:
                                continue
                            for i in range(1, len(route)-2):
                                for j in range(i+1, len(route)-1):
                                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                                    old_dist = route_distance(route)
                                    new_dist = route_distance(new_route)
                                    if new_dist >= old_dist - 1e-12:
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
                    elif phase in ['oropt1', 'oropt2', 'oropt3']:
                        block_size = int(phase[-1])
                        for r_idx in longest_routes:
                            route = routes[r_idx]
                            if len(route) <= block_size + 2:
                                continue
                            # Try to remove a block of block_size consecutive customers
                            for start in range(1, len(route) - block_size - 1):
                                block = route[start:start+block_size]
                                temp_route = route[:start] + route[start+block_size:]
                                # Try to insert block into any position in the same or other routes
                                # But to keep bounded, only insert into best position in temp_route or other routes
                                # First try to insert back into temp_route (intra)
                                for pos in range(1, len(temp_route)):
                                    new_route = temp_route[:pos] + block + temp_route[pos:]
                                    dist = route_distance(new_route)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != r_idx)
                                    new_max = max(dist, other_max)
                                    if new_max < best_max - 1e-12:
                                        routes[r_idx] = new_route
                                        best_routes = [r[:] for r in routes]
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                                # Then try to insert into other routes (inter)
                                for r2_idx in range(truck_count):
                                    if r2_idx == r_idx:
                                        continue
                                    r2 = routes[r2_idx]
                                    for pos in range(1, len(r2)):
                                        new_r2 = r2[:pos] + block + r2[pos:]
                                        dist1 = route_distance(temp_route)
                                        dist2 = route_distance(new_r2)
                                        other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != r_idx and x != r2_idx)
                                        new_max = max(dist1, dist2, other_max)
                                        if new_max < best_max - 1e-12:
                                            routes[r_idx] = temp_route
                                            routes[r2_idx] = new_r2
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
                    elif phase == 'inter2opt':
                        for r1 in longest_routes:
                            route1 = routes[r1]
                            if len(route1) <= 2:
                                continue
                            for r2 in range(truck_count):
                                if r2 == r1:
                                    continue
                                route2 = routes[r2]
                                if len(route2) <= 2:
                                    continue
                                for i in range(1, len(route1)-1):
                                    for j in range(1, len(route2)-1):
                                        new_route1 = route1[:i] + route2[j:]
                                        new_route2 = route2[:j] + route1[i:]
                                        dist1 = route_distance(new_route1)
                                        dist2 = route_distance(new_route2)
                                        other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != r1 and x != r2)
                                        new_max = max(dist1, dist2, other_max)
                                        if new_max < best_max - 1e-12:
                                            routes[r1] = new_route1
                                            routes[r2] = new_route2
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
                    elif phase == 'relocate':
                        for src in longest_routes:
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
                        for t1 in longest_routes:
                            route1 = routes[t1]
                            if len(route1) <= 2:
                                continue
                            for t2 in range(truck_count):
                                if t2 == t1:
                                    continue
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
                        for t1 in longest_routes:
                            route1 = routes[t1]
                            if len(route1) <= 2:
                                continue
                            for t2 in range(truck_count):
                                if t2 == t1:
                                    continue
                                route2 = routes[t2]
                                if len(route2) <= 2:
                                    continue
                                for i in range(1, len(route1)-1):
                                    for j in range(1, len(route2)-1):
                                        new_route1 = route1[:i] + route2[j:]
                                        new_route2 = route2[:j] + route1[i:]
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
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= 10:
                        break
            
            # Shake: remove a random contiguous segment from the longest route
            longest_idx = max(range(truck_count), key=lambda i: route_distance(routes[i]))
            longest_route = routes[longest_idx]
            if len(longest_route) > 4:  # at least 2 internal customers
                max_seg_len = min(3, len(longest_route) - 2)
                seg_len = random.randint(1, max_seg_len)
                start = random.randint(1, len(longest_route) - seg_len - 1)
                segment = longest_route[start:start+seg_len]
                new_longest = longest_route[:start] + longest_route[start+seg_len:]
                routes[longest_idx] = new_longest
                # Reinsert removed customers using regret
                removed = list(segment)
                for cust in removed:
                    best_increase = float('inf')
                    best_route = -1
                    best_pos = -1
                    current_max = max_distance(routes)
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
                            if increase < best_increase - 1e-12 or (abs(increase - best_increase) < 1e-12 and (r_idx < best_route or (r_idx == best_route and pos < best_pos))):
                                best_increase = increase
                                best_route = r_idx
                                best_pos = pos
                    routes[best_route].insert(best_pos, cust)
                cur_max = max_distance(routes)
                if cur_max < best_max - 1e-12:
                    best_routes = [r[:] for r in routes]
                    best_max = cur_max
                    report_best_vrp(best_routes)
        
        # update global best
        if best_max < best_global_max - 1e-12:
            best_global_max = best_max
            best_global_routes = [r[:] for r in best_routes]
            report_best_vrp(best_global_routes)
    
    if best_global_routes is None:
        best_global_routes = [r[:] for r in routes]
    return best_global_routes