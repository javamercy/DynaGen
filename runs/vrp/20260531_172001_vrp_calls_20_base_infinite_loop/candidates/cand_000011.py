import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # Initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    
    def route_length(route):
        total = 0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    # Regret-2 insertion
    unassigned = set(customers)
    while unassigned:
        best_customer = None
        best_regret = -1
        best_cost = None
        best_route_idx = None
        best_pos = None
        for cust in unassigned:
            costs = []
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    next_node = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, next_node] - distance_matrix[prev, next_node]
                    costs.append((cost, r_idx, pos))
            costs.sort(key=lambda x: x[0])
            best = costs[0][0]
            second = costs[1][0] if len(costs) > 1 else best + 1e9
            regret = second - best
            # Tie-breaking: larger regret, then smaller best cost, then smaller customer index
            if (best_customer is None or regret > best_regret or
                (regret == best_regret and (best_cost is None or best < best_cost)) or
                (regret == best_regret and best == best_cost and cust < best_customer)):
                best_regret = regret
                best_customer = cust
                best_cost = best
                best_route_idx = costs[0][1]
                best_pos = costs[0][2]
        # Insert
        routes[best_route_idx].insert(best_pos, best_customer)
        unassigned.remove(best_customer)
    
    # Report initial solution
    best_max = max(route_length(r) for r in routes)
    report_best_vrp(routes)
    
    # Intra-route 2-opt improvement
    max_iter = n * truck_count  # bound
    iteration = 0
    improved = True
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            best_route = route[:]
            best_len = route_length(route)
            improved_route = False
            for i in range(1, len(route)-2):
                for k in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                    new_len = route_length(new_route)
                    if new_len < best_len:
                        best_len = new_len
                        best_route = new_route
                        improved_route = True
            if improved_route:
                routes[r_idx] = best_route
                improved = True
                # Check max distance
                new_max = max(route_length(r) for r in routes)
                if new_max < best_max:
                    best_max = new_max
                    report_best_vrp(routes)
    
    return routes