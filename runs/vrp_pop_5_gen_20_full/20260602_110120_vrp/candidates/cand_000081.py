import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    random.seed(0)
    
    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def construction():
        routes = [[0, 0] for _ in range(truck_count)]
        route_distances = [0.0 for _ in range(truck_count)]
        unassigned = set(range(1, n))
        
        def best_max_position(customer):
            best_val = float('inf')
            best_pos = None
            second_val = float('inf')
            for r_idx, route in enumerate(routes):
                curr_dist = route_distances[r_idx]
                for i in range(1, len(route)):
                    new_dist = curr_dist - distance_matrix[route[i-1], route[i]] \
                               + distance_matrix[route[i-1], customer] \
                               + distance_matrix[customer, route[i]]
                    other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0.0)
                    cand_max = max(new_dist, other_max)
                    if cand_max < best_val - 1e-12:
                        second_val = best_val
                        best_val = cand_max
                        best_pos = (r_idx, i)
                    elif cand_max < second_val - 1e-12 and abs(cand_max - best_val) > 1e-12:
                        second_val = cand_max
            if best_pos is None:
                return float('inf'), float('inf'), (0, len(routes[0])-1)
            return best_val, second_val, best_pos
        
        while unassigned:
            regrets = []
            for c in unassigned:
                best_val, second_val, _ = best_max_position(c)
                regret = second_val - best_val if second_val != float('inf') else 0.0
                regrets.append((regret, best_val, c))
            regrets.sort(key=lambda x: (-x[0], -x[1], x[2]))
            selected = regrets[0][2]
            _, _, best_pos = best_max_position(selected)
            r_idx, i = best_pos
            route = routes[r_idx]
            route.insert(i, selected)
            route_distances[r_idx] = compute_route_distance(route)
            unassigned.remove(selected)
        return routes, route_distances
    
    def local_search(routes, route_distances):
        current_routes = [list(r) for r in routes]
        current_distances = list(route_distances)
        current_max = max(current_distances)
        improved = True
        max_iters = n * n
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            # Intra-route 2-opt
            for r_idx in range(truck_count):
                route = current_routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_route_distance(new_route)
                        if new_dist < current_distances[r_idx] - 1e-12:
                            current_distances[r_idx] = new_dist
                            current_routes[r_idx] = new_route
                            new_max = max(current_distances)
                            if new_max < current_max - 1e-12:
                                current_max = new_max
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Inter-swap
            for r1 in range(truck_count):
                for r2 in range(r1+1, truck_count):
                    route1 = current_routes[r1]
                    route2 = current_routes[r2]
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i] + [route2[j]] + route1[i+1:]
                            new2 = route2[:j] + [route1[i]] + route2[j+1:]
                            new_dist1 = compute_route_distance(new1)
                            new_dist2 = compute_route_distance(new2)
                            other_max = max(current_distances[k] for k in range(truck_count) if k not in (r1, r2))
                            cand_max = max(new_dist1, new_dist2, other_max)
                            if cand_max < current_max - 1e-12:
                                current_routes[r1] = new1
                                current_routes[r2] = new2
                                current_distances[r1] = new_dist1
                                current_distances[r2] = new_dist2
                                current_max = cand_max
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
            # Inter-relocate
            for r1 in range(truck_count):
                for r2 in range(truck_count):
                    if r1 == r2:
                        continue
                    route1 = current_routes[r1]
                    route2 = current_routes[r2]
                    for i in range(1, len(route1)-1):
                        c = route1[i]
                        new1 = route1[:i] + route1[i+1:]
                        new_dist1 = compute_route_distance(new1)
                        best_new2 = None
                        best_new_dist2 = float('inf')
                        for j in range(1, len(route2)):
                            new2 = route2[:j] + [c] + route2[j:]
                            d2 = compute_route_distance(new2)
                            if d2 < best_new_dist2:
                                best_new_dist2 = d2
                                best_new2 = new2
                        other_max = max([current_distances[k] for k in range(truck_count) if k not in (r1, r2)] or [0])
                        cand_max = max(new_dist1, best_new_dist2, other_max)
                        if cand_max < current_max - 1e-12:
                            current_routes[r1] = new1
                            current_routes[r2] = best_new2
                            current_distances[r1] = new_dist1
                            current_distances[r2] = best_new_dist2
                            current_max = cand_max
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return current_routes, current_distances, current_max
    
    def ruin_recreate(routes, route_distances):
        # Identify top two longest routes
        dist_with_idx = [(route_distances[i], i) for i in range(truck_count)]
        dist_with_idx.sort(reverse=True, key=lambda x: x[0])
        top_indices = [idx for _, idx in dist_with_idx[:2]]
        # Determine total customers to remove (30% of total customers)
        total_customers = n - 1
        total_remove = int(0.3 * total_customers)
        if total_remove <= 0:
            return routes, route_distances
        # Get removable customer indices per selected route
        removal_map = {}
        for idx in top_indices:
            route = routes[idx]
            interior = list(range(1, len(route)-1))
            if len(interior) > 0:
                removal_map[idx] = interior
            else:
                removal_map[idx] = []
        # Compute total available removals
        total_available = sum(len(v) for v in removal_map.values())
        if total_available == 0:
            return routes, route_distances
        if total_remove > total_available:
            total_remove = total_available
        # Distribute removal proportionally to route distances
        total_top_dist = sum(route_distances[i] for i in top_indices if i in removal_map)
        if total_top_dist == 0:
            total_top_dist = 1.0
        # Build list of (index, customer) to remove
        remove_list = []
        for idx in top_indices:
            interior = removal_map[idx]
            if not interior:
                continue
            prop = route_distances[idx] / total_top_dist
            num_remove = max(1, int(round(prop * total_remove)))
            num_remove = min(num_remove, len(interior))
            random.shuffle(interior)
            selected = interior[:num_remove]
            for cust in selected:
                remove_list.append((idx, cust))
        # Adjust if overshoot or undershoot
        if len(remove_list) > total_remove:
            remove_list = remove_list[:total_remove]
        elif len(remove_list) < total_remove:
            # Add from routes that still have removals left
            for idx in top_indices:
                interior = removal_map[idx]
                already_removed = [c for (i,c) in remove_list if i == idx]
                remaining = [c for c in interior if c not in already_removed]
                if remaining:
                    needed = total_remove - len(remove_list)
                    add = random.sample(remaining, min(needed, len(remaining)))
                    for c in add:
                        remove_list.append((idx, c))
                    if len(remove_list) >= total_remove:
                        break
        # Remove customers in reverse order to avoid index shifting
        remove_list.sort(key=lambda x: (x[0], -removal_map[x[0]].index(x[1]) if x[1] in removal_map[x[0]] else 0))
        new_routes = [list(r) for r in routes]
        new_distances = list(route_distances)
        removed_customers = []
        for idx, cust in reversed(remove_list):
            route = new_routes[idx]
            route.remove(cust)  # only interior, so safe
            removed_customers.append(cust)
        # Recompute distances for affected routes
        for idx in set(idx for idx,_ in remove_list):
            new_distances[idx] = compute_route_distance(new_routes[idx])
        # Recreate: cheapest insertion minimizing max distance
        unassigned = set(removed_customers)
        while unassigned:
            best_customer = None
            best_max_val = float('inf')
            best_pos = None
            for c in unassigned:
                for r_idx, r in enumerate(new_routes):
                    for i in range(1, len(r)):
                        new_dist = new_distances[r_idx] - distance_matrix[r[i-1], r[i]] \
                                   + distance_matrix[r[i-1], c] \
                                   + distance_matrix[c, r[i]]
                        other_max = max(new_distances[:r_idx] + new_distances[r_idx+1:], default=0.0)
                        cand_max = max(new_dist, other_max)
                        if cand_max < best_max_val - 1e-12:
                            best_max_val = cand_max
                            best_customer = c
                            best_pos = (r_idx, i)
            # Insert best
            r_idx, i = best_pos
            new_routes[r_idx].insert(i, best_customer)
            new_distances[r_idx] = compute_route_distance(new_routes[r_idx])
            unassigned.remove(best_customer)
        return new_routes, new_distances
    
    best_routes = None
    best_max_val = float('inf')
    max_restarts = min(10, n) if n > 0 else 1
    
    routes, route_distances = construction()
    routes, route_distances, current_max = local_search(routes, route_distances)
    report_best_vrp(routes)
    if current_max < best_max_val - 1e-12:
        best_max_val = current_max
        best_routes = routes
    
    for restart in range(1, max_restarts):
        routes, route_distances = ruin_recreate(best_routes, [compute_route_distance(r) for r in best_routes])
        routes, route_distances, current_max = local_search(routes, route_distances)
        if current_max < best_max_val - 1e-12:
            best_max_val = current_max
            best_routes = routes
            report_best_vrp(best_routes)
    
    return best_routes