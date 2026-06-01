import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    # Initialize empty routes
    routes = [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    # Construction: cheapest insertion
    for cust in customers:
        best_truck, best_pos, best_inc = -1, -1, float('inf')
        for t, route in enumerate(routes):
            for i in range(len(route)-1):
                inc = distance_matrix[route[i]][cust] + distance_matrix[cust][route[i+1]] - distance_matrix[route[i]][route[i+1]]
                if inc < best_inc:
                    best_inc = inc
                    best_truck = t
                    best_pos = i+1
        routes[best_truck].insert(best_pos, cust)
    
    def route_distance(route):
        total = 0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total
    
    distances = [route_distance(r) for r in routes]
    best_max = max(distances)
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)
    
    # Simulated annealing parameters
    initial_T = 0.5 * best_max if best_max > 0 else 1.0
    max_iter = (n - 1) * truck_count * 10
    T = initial_T
    num_random_moves = 100  # evaluate this many random moves per iteration
    
    for iteration in range(max_iter):
        # Generate random moves
        candidate_moves = []
        for _ in range(num_random_moves):
            move_type = random.choice(['relocate', 'swap', '2opt'])
            if move_type == 'relocate':
                t1 = random.randrange(truck_count)
                if len(routes[t1]) <= 2:
                    continue
                idx1 = random.randrange(1, len(routes[t1])-1)
                t2 = random.randrange(truck_count)
                if t2 == t1:
                    continue
                idx2 = random.randrange(len(routes[t2])-1)
                insert_pos = idx2+1
                candidate_moves.append(('relocate', t1, idx1, t2, insert_pos))
            elif move_type == 'swap':
                t1 = random.randrange(truck_count)
                if len(routes[t1]) <= 2:
                    continue
                idx1 = random.randrange(1, len(routes[t1])-1)
                t2 = random.randrange(truck_count)
                if t2 == t1:
                    continue
                if len(routes[t2]) <= 2:
                    continue
                idx2 = random.randrange(1, len(routes[t2])-1)
                candidate_moves.append(('swap', t1, idx1, t2, idx2))
            else:  # 2opt
                t = random.randrange(truck_count)
                if len(routes[t]) <= 3:
                    continue
                i = random.randrange(1, len(routes[t])-2)
                j = random.randrange(i+1, len(routes[t])-1)
                candidate_moves.append(('2opt', t, i, j))
        
        # Evaluate candidate moves and find best improving
        best_move = None
        best_new_max = best_max
        for move in candidate_moves:
            if move[0] == 'relocate':
                _, t1, idx1, t2, pos = move
                cust = routes[t1][idx1]
                new_route1 = routes[t1][:idx1] + routes[t1][idx1+1:]
                new_route2 = routes[t2][:pos] + [cust] + routes[t2][pos:]
                new_dist1 = route_distance(new_route1)
                new_dist2 = route_distance(new_route2)
                new_distances = distances.copy()
                new_distances[t1] = new_dist1
                new_distances[t2] = new_dist2
                new_max = max(new_distances)
            elif move[0] == 'swap':
                _, t1, idx1, t2, idx2 = move
                cust1 = routes[t1][idx1]
                cust2 = routes[t2][idx2]
                new_route1 = routes[t1][:idx1] + [cust2] + routes[t1][idx1+1:]
                new_route2 = routes[t2][:idx2] + [cust1] + routes[t2][idx2+1:]
                new_dist1 = route_distance(new_route1)
                new_dist2 = route_distance(new_route2)
                new_distances = distances.copy()
                new_distances[t1] = new_dist1
                new_distances[t2] = new_dist2
                new_max = max(new_distances)
            else:  # 2opt
                _, t, i, j = move
                new_route = routes[t][:i] + routes[t][i:j+1][::-1] + routes[t][j+1:]
                new_dist = route_distance(new_route)
                new_distances = distances.copy()
                new_distances[t] = new_dist
                new_max = max(new_distances)
            if new_max < best_new_max:
                best_new_max = new_max
                best_move = move
        
        # Accept best improving move if found
        if best_move is not None and best_new_max < best_max:
            # Apply best move
            if best_move[0] == 'relocate':
                _, t1, idx1, t2, pos = best_move
                cust = routes[t1][idx1]
                del routes[t1][idx1]
                routes[t2].insert(pos, cust)
            elif best_move[0] == 'swap':
                _, t1, idx1, t2, idx2 = best_move
                routes[t1][idx1], routes[t2][idx2] = routes[t2][idx2], routes[t1][idx1]
            else:  # 2opt
                _, t, i, j = best_move
                routes[t] = routes[t][:i] + routes[t][i:j+1][::-1] + routes[t][j+1:]
            distances = [route_distance(r) for r in routes]
            current_max = max(distances)
            if current_max < best_max:
                best_max = current_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
        else:
            # No improving move; try a random move with SA acceptance
            if candidate_moves:
                move = random.choice(candidate_moves)
                # Compute new max if move applied
                if move[0] == 'relocate':
                    _, t1, idx1, t2, pos = move
                    cust = routes[t1][idx1]
                    new_route1 = routes[t1][:idx1] + routes[t1][idx1+1:]
                    new_route2 = routes[t2][:pos] + [cust] + routes[t2][pos:]
                    new_dist1 = route_distance(new_route1)
                    new_dist2 = route_distance(new_route2)
                    new_distances = distances.copy()
                    new_distances[t1] = new_dist1
                    new_distances[t2] = new_dist2
                    new_max = max(new_distances)
                elif move[0] == 'swap':
                    _, t1, idx1, t2, idx2 = move
                    cust1 = routes[t1][idx1]
                    cust2 = routes[t2][idx2]
                    new_route1 = routes[t1][:idx1] + [cust2] + routes[t1][idx1+1:]
                    new_route2 = routes[t2][:idx2] + [cust1] + routes[t2][idx2+1:]
                    new_dist1 = route_distance(new_route1)
                    new_dist2 = route_distance(new_route2)
                    new_distances = distances.copy()
                    new_distances[t1] = new_dist1
                    new_distances[t2] = new_dist2
                    new_max = max(new_distances)
                else:  # 2opt
                    _, t, i, j = move
                    new_route = routes[t][:i] + routes[t][i:j+1][::-1] + routes[t][j+1:]
                    new_dist = route_distance(new_route)
                    new_distances = distances.copy()
                    new_distances[t] = new_dist
                    new_max = max(new_distances)
                delta = new_max - best_max
                if delta < 0 or random.random() < math.exp(-delta / T):
                    # Accept move
                    if move[0] == 'relocate':
                        _, t1, idx1, t2, pos = move
                        cust = routes[t1][idx1]
                        del routes[t1][idx1]
                        routes[t2].insert(pos, cust)
                    elif move[0] == 'swap':
                        _, t1, idx1, t2, idx2 = move
                        routes[t1][idx1], routes[t2][idx2] = routes[t2][idx2], routes[t1][idx1]
                    else:
                        _, t, i, j = move
                        routes[t] = new_route
                    distances = [route_distance(r) for r in routes]
                    current_max = max(distances)
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
        # Update temperature
        T = initial_T * (1 - (iteration+1)/max_iter)
    return best_routes