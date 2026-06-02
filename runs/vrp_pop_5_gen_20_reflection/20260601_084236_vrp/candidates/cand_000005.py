import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    def report_best_vrp(routes):
        pass
    
    # Construction: greedy insertion minimizing max distance
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))
    while unassigned:
        best_max = float('inf')
        best_ins = None
        for cust in sorted(unassigned):
            for t in range(truck_count):
                route = routes[t]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_routes = routes.copy()
                    new_routes[t] = new_route
                    new_max = max_distance(new_routes)
                    if new_max < best_max:
                        best_max = new_max
                        best_ins = (cust, t, pos)
        if best_ins:
            cust, t, pos = best_ins
            routes[t] = routes[t][:pos] + [cust] + routes[t][pos:]
            unassigned.remove(cust)
        else:
            break
    
    best_routes = [list(r) for r in routes]
    best_max = max_distance(best_routes)
    report_best_vrp(best_routes)
    
    # Improvement
    max_iter = n * truck_count * 10
    for _ in range(max_iter):
        improved = False
        # Relocate
        for t1 in range(truck_count):
            if len(routes[t1]) < 3:
                continue
            for pos1 in range(1, len(routes[t1])-1):
                cust = routes[t1][pos1]
                new_route_t1 = routes[t1][:pos1] + routes[t1][pos1+1:]
                for t2 in range(truck_count):
                    route_t2 = routes[t2] if t2 != t1 else new_route_t1
                    for pos2 in range(1, len(route_t2)+1):
                        new_route_t2 = route_t2[:pos2] + [cust] + route_t2[pos2:]
                        new_routes = routes.copy()
                        new_routes[t1] = new_route_t1
                        new_routes[t2] = new_route_t2
                        new_max = max_distance(new_routes)
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = [list(r) for r in new_routes]
                            report_best_vrp(best_routes)
                            routes = new_routes
                            improved = True
                            break
                    if improved: break
                if improved: break
            if improved: break
        if improved: continue
        # Swap
        for t1 in range(truck_count):
            if len(routes[t1]) < 3:
                continue
            for pos1 in range(1, len(routes[t1])-1):
                cust1 = routes[t1][pos1]
                for t2 in range(t1+1, truck_count):
                    if len(routes[t2]) < 3:
                        continue
                    for pos2 in range(1, len(routes[t2])-1):
                        cust2 = routes[t2][pos2]
                        new_routes = [list(r) for r in routes]
                        new_routes[t1][pos1] = cust2
                        new_routes[t2][pos2] = cust1
                        new_max = max_distance(new_routes)
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = [list(r) for r in new_routes]
                            report_best_vrp(best_routes)
                            routes = new_routes
                            improved = True
                            break
                    if improved: break
                if improved: break
            if improved: break
        if improved: continue
        # 2-opt within routes
        for t in range(truck_count):
            if len(routes[t]) < 4:
                continue
            for i in range(1, len(routes[t])-2):
                for j in range(i+1, len(routes[t])-1):
                    new_route = routes[t][:i] + routes[t][i:j+1][::-1] + routes[t][j+1:]
                    new_routes = routes.copy()
                    new_routes[t] = new_route
                    new_max = max_distance(new_routes)
                    if new_max < best_max:
                        best_max = new_max
                        best_routes = [list(r) for r in new_routes]
                        report_best_vrp(best_routes)
                        routes = new_routes
                        improved = True
                        break
                if improved: break
            if improved: break
        if not improved:
            break
    
    # Ensure correct number of routes
    while len(best_routes) < truck_count:
        best_routes.append([0,0])
    return best_routes