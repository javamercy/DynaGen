import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]
    
    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    def max_route_len(routes):
        return max(route_length(r) for r in routes) if routes else float('inf')
    
    def total_route_len(routes):
        return sum(route_length(r) for r in routes)
    
    def construct_routes(randomize=False):
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        while unassigned:
            candidates = []
            for cust in unassigned:
                insert_info = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_len = route_length(route) + cost
                        other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                        new_max = max(new_len, *other_lens)
                        insert_info.append((new_max, cost, r_idx, pos))
                insert_info.sort(key=lambda x: (x[0], x[1]))
                best = insert_info[0]
                second = insert_info[1] if len(insert_info) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                regret = second[0] - best[0]
                candidates.append((best[0], regret, best[1], best[2], best[3], cust))
            candidates.sort(key=lambda x: (x[0], -x[1], -x[2], x[5]))
            if randomize and len(candidates) >= 2:
                chosen = random.choice([candidates[0], candidates[1]])
            else:
                chosen = candidates[0]
            _, _, _, r_idx, pos, cust = chosen
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)
        return routes
    
    pop_size = max(10, n // 5)
    generations = max(20, n * truck_count)
    mutation_rate = 0.1
    
    population = []
    for _ in range(pop_size):
        routes = construct_routes(randomize=True)
        population.append(routes)
    
    fitness = [(max_route_len(r), total_route_len(r), r) for r in population]
    best_fitness = min(fitness, key=lambda x: (x[0], x[1]))
    best_routes = best_fitness[2][:]
    report_best_vrp(best_routes)
    
    for gen in range(generations):
        i1 = random.randint(0, pop_size-1)
        i2 = random.randint(0, pop_size-1)
        parent1 = population[i1] if (fitness[i1][0], fitness[i1][1]) < (fitness[i2][0], fitness[i2][1]) else population[i2]
        i1 = random.randint(0, pop_size-1)
        i2 = random.randint(0, pop_size-1)
        parent2 = population[i1] if (fitness[i1][0], fitness[i1][1]) < (fitness[i2][0], fitness[i2][1]) else population[i2]
        
        child = []
        assigned = set()
        for route in parent1:
            if random.random() < 0.5:
                custs = route[1:-1]
                if not any(c in assigned for c in custs):
                    child.append(route[:])
                    assigned.update(custs)
        while len(child) < truck_count:
            child.append([0, 0])
        remaining_custs = [c for c in range(1, n) if c not in assigned]
        unassigned = set(remaining_custs)
        while unassigned:
            best_cust = None
            best_regret = -1.0
            best_data = None
            for cust in unassigned:
                insert_info = []
                for r_idx, route in enumerate(child):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_len = route_length(route) + cost
                        other_lens = [route_length(child[i]) for i in range(truck_count) if i != r_idx]
                        new_max = max(new_len, *other_lens)
                        insert_info.append((new_max, cost, r_idx, pos))
                if not insert_info:
                    continue
                insert_info.sort(key=lambda x: (x[0], x[1]))
                best = insert_info[0]
                second = insert_info[1] if len(insert_info) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                regret = second[0] - best[0]
                if best_cust is None or regret > best_regret or (regret == best_regret and best[1] < best_data[1]):
                    best_cust = cust
                    best_regret = regret
                    best_data = (best[0], best[1], best[2], best[3])
            if best_cust is None:
                break
            _, _, r_idx, pos = best_data
            child[r_idx].insert(pos, best_cust)
            unassigned.remove(best_cust)
        
        if random.random() < mutation_rate:
            custs_list = [c for c in range(1, n)]
            c1, c2 = random.sample(custs_list, 2)
            r1_idx, pos1, r2_idx, pos2 = None, None, None, None
            for r_idx, route in enumerate(child):
                for pos, c in enumerate(route):
                    if c == c1:
                        r1_idx, pos1 = r_idx, pos
                    if c == c2:
                        r2_idx, pos2 = r_idx, pos
            child[r1_idx][pos1] = c2
            child[r2_idx][pos2] = c1
        
        child_max = max_route_len(child)
        child_total = total_route_len(child)
        # find worst
        worst_idx = 0
        worst_val = (fitness[0][0], fitness[0][1])
        for i in range(1, pop_size):
            if (fitness[i][0], fitness[i][1]) > worst_val:
                worst_val = (fitness[i][0], fitness[i][1])
                worst_idx = i
        if (child_max, child_total) < (fitness[worst_idx][0], fitness[worst_idx][1]):
            population[worst_idx] = child
            fitness[worst_idx] = (child_max, child_total, child)
            if (child_max, child_total) < (best_fitness[0], best_fitness[1]):
                best_fitness = (child_max, child_total, child)
                best_routes = [r[:] for r in child]
                report_best_vrp(best_routes)
    
    return best_routes