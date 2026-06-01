import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0.0] * truck_count
        for cust in perm:
            best_max = float('inf')
            best_r = -1
            best_p = -1
            for r in range(truck_count):
                route = routes[r]
                for p in range(1, len(route)):
                    prev = route[p-1]
                    nxt = route[p]
                    new_len = lengths[r] - distance_matrix[prev, nxt] + distance_matrix[prev, cust] + distance_matrix[cust, nxt]
                    new_max = new_len
                    for rr in range(truck_count):
                        if rr != r and lengths[rr] > new_max:
                            new_max = lengths[rr]
                    if new_max < best_max:
                        best_max = new_max
                        best_r = r
                        best_p = p
            routes[best_r].insert(best_p, cust)
            lengths[best_r] = route_distance(routes[best_r])
        max_len = max(lengths)
        return routes, max_len
    
    def routes_to_perm(routes):
        perm = []
        for r in routes:
            for c in r[1:-1]:
                perm.append(c)
        return perm
    
    def local_search(routes, best_max_global):
        nonlocal best_routes_global, best_max_global
        route_dists = [route_distance(r) for r in routes]
        best_local_max = max(route_dists)
        best_local_routes = [list(r) for r in routes]
        tabu_tenure = 7
        tabu_list = []
        tabu_set = set()
        max_iter = min(50, n)
        iterations_since_improvement = 0
        improvement_threshold = max(5, n // 5)
        for _ in range(max_iter):
            best_move = None
            best_new_max = float('inf')
            best_tie = None
            # Relocate
            for t1 in range(truck_count):
                route1 = routes[t1]
                if len(route1) <= 2:
                    continue
                for i in range(1, len(route1)-1):
                    cust = route1[i]
                    new_route1 = route1[:i] + route1[i+1:]
                    dist1_new = route_distance(new_route1)
                    for t2 in range(truck_count):
                        if t1 == t2:
                            continue
                        route2 = routes[t2]
                        for j in range(1, len(route2)):
                            new_route2 = route2[:j] + [cust] + route2[j:]
                            dist2_new = route_distance(new_route2)
                            new_max = max(dist1_new, dist2_new)
                            for k in range(truck_count):
                                if k != t1 and k != t2:
                                    new_max = max(new_max, route_dists[k])
                            is_tabu = (cust, t2, t1) in tabu_set
                            if is_tabu and new_max >= best_local_max:
                                continue
                            tie = (new_max, 0, t1, i, t2, j)
                            if best_tie is None or tie < best_tie:
                                best_new_max = new_max
                                best_move = ('relocate', t1, i, t2, j, cust)
                                best_tie = tie
            # Swap
            for t1 in range(truck_count):
                route1 = routes[t1]
                if len(route1) <= 2:
                    continue
                for i in range(1, len(route1)-1):
                    cust1 = route1[i]
                    for t2 in range(t1+1, truck_count):
                        route2 = routes[t2]
                        if len(route2) <= 2:
                            continue
                        for j in range(1, len(route2)-1):
                            cust2 = route2[j]
                            new_route1 = route1[:i] + [cust2] + route1[i+1:]
                            new_route2 = route2[:j] + [cust1] + route2[j+1:]
                            dist1_new = route_distance(new_route1)
                            dist2_new = route_distance(new_route2)
                            new_max = max(dist1_new, dist2_new)
                            for k in range(truck_count):
                                if k != t1 and k != t2:
                                    new_max = max(new_max, route_dists[k])
                            is_tabu = ((cust1, cust2, t1, t2) in tabu_set) or ((cust2, cust1, t2, t1) in tabu_set)
                            if is_tabu and new_max >= best_local_max:
                                continue
                            tie = (new_max, 1, t1, i, t2, j)
                            if best_tie is None or tie < best_tie:
                                best_new_max = new_max
                                best_move = ('swap', t1, i, t2, j, cust1, cust2)
                                best_tie = tie
            # 2-opt
            for t in range(truck_count):
                route = routes[t]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        new_max = new_dist
                        for k in range(truck_count):
                            if k != t:
                                new_max = max(new_max, route_dists[k])
                        tie = (new_max, 2, t, i, j)
                        if best_tie is None or tie < best_tie:
                            best_new_max = new_max
                            best_move = ('2opt', t, i, j, new_route)
                            best_tie = tie
            if best_move is None:
                break
            # Apply move
            if best_move[0] == 'relocate':
                _, t1, i, t2, j, cust = best_move
                routes[t1] = routes[t1][:i] + routes[t1][i+1:]
                routes[t2] = routes[t2][:j] + [cust] + routes[t2][j:]
                route_dists[t1] = route_distance(routes[t1])
                route_dists[t2] = route_distance(routes[t2])
                tabu_entry = (cust, t2, t1)
                tabu_list.append(tabu_entry)
                tabu_set.add(tabu_entry)
            elif best_move[0] == 'swap':
                _, t1, i, t2, j, cust1, cust2 = best_move
                routes[t1][i] = cust2
                routes[t2][j] = cust1
                route_dists[t1] = route_distance(routes[t1])
                route_dists[t2] = route_distance(routes[t2])
                tabu_entry1 = (cust1, cust2, t1, t2)
                tabu_entry2 = (cust2, cust1, t2, t1)
                tabu_list.append(tabu_entry1)
                tabu_set.add(tabu_entry1)
                tabu_list.append(tabu_entry2)
                tabu_set.add(tabu_entry2)
            else:
                _, t, i, j, new_route = best_move
                routes[t] = new_route
                route_dists[t] = route_distance(new_route)
            # Manage tabu list
            while len(tabu_list) > tabu_tenure:
                entry = tabu_list.pop(0)
                tabu_set.discard(entry)
            # Check improvement
            current_max = max(route_dists)
            if current_max < best_local_max:
                best_local_max = current_max
                best_local_routes = [list(r) for r in routes]
                iterations_since_improvement = 0
                if tabu_tenure > 3:
                    tabu_tenure -= 1
                if current_max < best_max_global:
                    best_max_global = current_max
                    best_routes_global = [list(r) for r in routes]
                    report_best_vrp(best_routes_global)
            else:
                iterations_since_improvement += 1
                if iterations_since_improvement >= improvement_threshold and tabu_tenure < 12:
                    tabu_tenure += 1
                    iterations_since_improvement = 0
        return best_local_routes, best_local_max
    
    # Regret-based construction to get one good permutation
    def regret_construction():
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = set(customers)
        while unassigned:
            best_regret = -1.0
            selected_cust = None
            best_info = None
            for cust in list(unassigned):
                best_new_max = float('inf')
                second_best_new_max = float('inf')
                best_t = -1
                best_pos = -1
                for t in range(truck_count):
                    route = routes[t]
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_len = route_distance(new_route)
                        new_max = new_len
                        for k in range(truck_count):
                            if k != t:
                                new_max = max(new_max, route_dists[k])
                        if new_max < best_new_max:
                            second_best_new_max = best_new_max
                            best_new_max = new_max
                            best_t = t
                            best_pos = pos
                        elif new_max < second_best_new_max:
                            second_best_new_max = new_max
                regret = second_best_new_max - best_new_max
                if regret > best_regret or (regret == best_regret and cust < selected_cust):
                    best_regret = regret
                    selected_cust = cust
                    best_info = (best_t, best_pos)
            t, pos = best_info
            routes[t] = routes[t][:pos] + [selected_cust] + routes[t][pos:]
            route_dists[t] = route_distance(routes[t])
            unassigned.remove(selected_cust)
        return routes_to_perm(routes)
    
    best_routes_global = None
    best_max_global = float('inf')
    def report_best_vrp(routes):
        nonlocal best_routes_global, best_max_global
        m = max(route_distance(r) for r in routes)
        if m < best_max_global:
            best_max_global = m
            best_routes_global = [list(r) for r in routes]
    
    # Parameters
    pop_size = min(30, n)
    max_gen = 2 * n
    mutation_prob = 0.1
    
    # Initialize population
    population = []
    regret_perm = regret_construction()
    for _ in range(pop_size):
        if _ == 0:
            perm = regret_perm[:]
        else:
            perm = customers[:]
            random.shuffle(perm)
        routes, max_len = decode(perm)
        report_best_vrp(routes)
        population.append((max_len, perm))
    population.sort(key=lambda x: x[0])
    
    # Steady-state genetic algorithm with local search
    for _ in range(max_gen):
        # Binary tournament selection
        idx1 = random.sample(range(pop_size), 2)
        idx2 = random.sample(range(pop_size), 2)
        parent1 = population[idx1[0]] if population[idx1[0]][0] <= population[idx1[1]][0] else population[idx1[1]]
        parent2 = population[idx2[0]] if population[idx2[0]][0] <= population[idx2[1]][0] else population[idx2[1]]
        p1 = parent1[1]
        p2 = parent2[1]
        
        # PMX crossover
        n_cust = len(customers)
        a = random.randint(0, n_cust-1)
        b = random.randint(0, n_cust-1)
        if a > b:
            a, b = b, a
        child = [None] * n_cust
        child[a:b+1] = p1[a:b+1]
        mapping = {}
        for i in range(a, b+1):
            mapping[p1[i]] = p2[i]
        for i in range(n_cust):
            if i < a or i > b:
                gene = p2[i]
                while gene in mapping:
                    gene = mapping[gene]
                child[i] = gene
        used = set(child)
        remaining = [c for c in customers if c not in used]
        for i in range(n_cust):
            if child[i] is None:
                child[i] = remaining.pop()
        
        # Mutation
        if random.random() < mutation_prob:
            i = random.randint(0, n_cust-1)
            j = random.randint(0, n_cust-1)
            child[i], child[j] = child[j], child[i]
        
        # Decode child
        routes_child, max_child = decode(child)
        report_best_vrp(routes_child)
        
        # Local search on child's routes
        routes_improved, max_improved = local_search(routes_child, best_max_global)
        if max_improved < max_child:
            max_child = max_improved
            # Convert improved routes back to permutation (order of customers)
            child = routes_to_perm(routes_improved)
        
        # Replace worst if child is better
        if max_child < population[-1][0]:
            population[-1] = (max_child, child)
            population.sort(key=lambda x: x[0])
    
    return best_routes_global