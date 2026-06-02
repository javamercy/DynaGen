import numpy as np
from itertools import permutations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix.tolist()
    customers = list(range(1, n))
    
    # Construction: minimax insertion (farthest customers first)
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    customers_sorted = sorted(customers, key=lambda c: dist[0][c], reverse=True)
    for c in customers_sorted:
        best_max = float('inf')
        best_inc = float('inf')
        best_r = -1
        best_pos = -1
        for r in range(truck_count):
            route = routes[r]
            cur_dist = route_dists[r]
            for i in range(len(route) - 1):
                a = route[i]
                b = route[i+1]
                inc = dist[a][c] + dist[c][b] - dist[a][b]
                new_dist = cur_dist + inc
                other_max = max(route_dists[:r] + route_dists[r+1:] + [0.0])
                new_max = max(other_max, new_dist)
                if (new_max < best_max) or (new_max == best_max and inc < best_inc):
                    best_max = new_max
                    best_inc = inc
                    best_r = r
                    best_pos = i
        routes[best_r].insert(best_pos + 1, c)
        route_dists[best_r] += best_inc
    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)
    
    def route_distance(route):
        return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))
    
    def objective(routes_list):
        return max(route_distance(r) for r in routes_list)
    
    # Helper: find customer position
    def find_customer(node, routes_list):
        for r_idx, route in enumerate(routes_list):
            for p, cust in enumerate(route):
                if cust == node:
                    return r_idx, p
        return None, None
    
    # Ruin-and-recreate (deterministic)
    num_customers = len(customers)
    ruin_size = max(1, int(0.2 * num_customers))
    max_iter = 10
    for iteration in range(max_iter):
        improved = False
        # Identify longest route
        max_dist = max(route_dists)
        long_routes = [r for r, d in enumerate(route_dists) if d == max_dist]
        target_route = long_routes[0]  # first longest
        
        # Collect customers from target route with high contribution (distance sum to neighbors)
        target_route_cust = [c for c in routes[target_route] if c != 0]
        contributions = []
        for c in target_route_cust:
            r, pos = find_customer(c, routes)
            if r is None:
                continue
            route = routes[r]
            left = route[pos-1] if pos > 0 else 0
            right = route[pos+1] if pos < len(route)-1 else 0
            contrib = dist[left][c] + dist[c][right]  # contribution to route distance
            contributions.append((contrib, c))
        contributions.sort(reverse=True, key=lambda x: (x[0], -x[1]))  # break ties by larger node? we want deterministic, so use node index descending for consistency? Actually we want highest contribution, then if tie smallest node? We'll do (x[0], -x[1]) to prioritize larger node? But to be deterministic, just sort by contrib descending then node ascending.
        # Let's use ( -contrib, node ) so ascending. But we need descending contrib. So we can sort by (-contrib, node).
        contributions.sort(key=lambda x: (-x[0], x[1]))
        
        # Remove from target route first (70% of ruin size)
        num_from_target = min(len(contributions), int(0.7 * ruin_size))
        removed_nodes = []
        for i in range(num_from_target):
            c = contributions[i][1]
            removed_nodes.append(c)
        # Remove remaining from other routes
        remaining = ruin_size - num_from_target
        if remaining > 0:
            other_cust = []
            for r_idx, route in enumerate(routes):
                if r_idx == target_route:
                    continue
                for c in route:
                    if c != 0 and c not in removed_nodes:
                        r, pos = find_customer(c, routes)
                        if r is None:
                            continue
                        left = route[pos-1] if pos > 0 else 0
                        right = route[pos+1] if pos < len(route)-1 else 0
                        contrib = dist[left][c] + dist[c][right]
                        other_cust.append((contrib, c, r_idx, pos))
            other_cust.sort(key=lambda x: (-x[0], x[1]))
            for i in range(min(remaining, len(other_cust))):
                c = other_cust[i][1]
                if c not in removed_nodes:
                    removed_nodes.append(c)
        
        # Remove these customers from routes
        for c in removed_nodes:
            r, pos = find_customer(c, routes)
            if r is not None and pos is not None:
                del routes[r][pos]
                if len(routes[r]) < 2:
                    routes[r] = [0, 0]
        route_dists = [route_distance(r) for r in routes]
        
        # Reinsert removed customers in descending order of depot distance (farthest first)
        removed_sorted = sorted(removed_nodes, key=lambda c: dist[0][c], reverse=True)
        for c in removed_sorted:
            best_max = float('inf')
            best_inc = float('inf')
            best_r = -1
            best_pos = -1
            for r in range(truck_count):
                route = routes[r]
                cur_dist = route_dists[r]
                for i in range(len(route) - 1):
                    a = route[i]
                    b = route[i+1]
                    inc = dist[a][c] + dist[c][b] - dist[a][b]
                    new_dist = cur_dist + inc
                    other_max = max(route_dists[:r] + route_dists[r+1:] + [0.0])
                    new_max = max(other_max, new_dist)
                    if (new_max < best_max) or (new_max == best_max and inc < best_inc):
                        best_max = new_max
                        best_inc = inc
                        best_r = r
                        best_pos = i
            routes[best_r].insert(best_pos + 1, c)
            route_dists[best_r] += best_inc
        
        current_max = max(route_dists)
        if current_max < best_max:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
            improved = True
        else:
            # revert to best solution?
            routes = [list(r) for r in best_routes]
            route_dists = [route_distance(r) for r in routes]
    
    # Final 2-opt local search
    for r_idx in range(truck_count):
        route = routes[r_idx]
        if len(route) > 3:
            for i in range(1, len(route)-2):
                for k in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < route_dists[r_idx]:
                        routes[r_idx] = new_route
                        route_dists[r_idx] = new_dist
                        current_max = max(route_dists)
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                        break
    
    if max(route_dists) < best_max:
        best_routes = [list(r) for r in routes]
    return best_routes