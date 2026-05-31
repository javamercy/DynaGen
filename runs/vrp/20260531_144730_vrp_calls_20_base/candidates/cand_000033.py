import numpy as np
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    n_cust = len(customers)
    if truck_count >= n_cust:
        routes = [[0, 0] for _ in range(truck_count)]
        for i, cust in enumerate(customers):
            routes[i] = [0, cust, 0]
        return routes

    # -------------------------------
    # Helper functions
    # -------------------------------
    def route_dist(route):
        if len(route) < 2:
            return 0.0
        d = 0.0
        for a, b in zip(route[:-1], route[1:]):
            d += distance_matrix[a, b]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    def two_opt(route):
        best = route[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best)-2):
                for j in range(i+1, len(best)-1):
                    if j - i == 1:
                        continue
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    if route_dist(new_route) < route_dist(best):
                        best = new_route
                        improved = True
                        break
                if improved:
                    break
        return best

    def split_sequence(seq):
        """Given a permutation of customers, split into exactly truck_count routes minimizing max distance, then apply 2-opt."""
        nc = len(seq)
        K = truck_count
        # Precompute segment distances
        seg = [[0.0] * nc for _ in range(nc)]
        for i in range(nc):
            d = distance_matrix[0, seq[i]]
            seg[i][i] = d + distance_matrix[seq[i], 0]
            for j in range(i+1, nc):
                d += distance_matrix[seq[j-1], seq[j]]
                seg[i][j] = d + distance_matrix[seq[j], 0]
        INF = 1e15
        dp = [[INF] * (nc + 1) for _ in range(K + 1)]
        parent = [[-1] * (nc + 1) for _ in range(K + 1)]
        dp[0][0] = 0.0
        for k in range(1, K+1):
            for i in range(k, nc+1):
                for j in range(k-1, i):
                    cand = max(dp[k-1][j], seg[j][i-1])
                    if cand < dp[k][i]:
                        dp[k][i] = cand
                        parent[k][i] = j
        # Reconstruct routes
        routes = []
        k = K
        i = nc
        while k > 0:
            j = parent[k][i]
            segment = seq[j:i]
            route = [0] + segment + [0]
            routes.append(route)
            i = j
            k -= 1
        routes.reverse()
        # Add empty routes if needed (should not happen because K <= nc)
        while len(routes) < truck_count:
            routes.append([0, 0])
        # Apply 2-opt to each route
        for idx in range(truck_count):
            routes[idx] = two_opt(routes[idx])
        return routes

    # -------------------------------
    # Initial population (deterministic)
    # -------------------------------
    # Method 1: Nearest neighbor starting from customer 1
    def nearest_neighbor_tour(start):
        unvisited = set(customers)
        tour = [start]
        unvisited.remove(start)
        current = start
        while unvisited:
            next_cust = min(unvisited, key=lambda x: distance_matrix[current, x])
            tour.append(next_cust)
            unvisited.remove(next_cust)
            current = next_cust
        return tour

    # Method 2: Cluster-first (from parent2) - convert to permutation
    def cluster_first_perm():
        # Farthest-first seeding
        centers = [np.argmax(distance_matrix[0, 1:]) + 1]
        for _ in range(1, truck_count):
            dist_to_centers = np.min([[distance_matrix[c, i] for i in range(1, n)] for c in centers], axis=0)
            new_center = np.argmax(dist_to_centers) + 1
            centers.append(new_center)
        centers = np.array(centers)
        # Assign each customer to nearest center
        clusters = [[] for _ in range(truck_count)]
        for cust in range(1, n):
            dists = [distance_matrix[center, cust] for center in centers]
            cluster_idx = np.argmin(dists)
            clusters[cluster_idx].append(cust)
        # Build routes via nearest neighbor and 2-opt
        routes = []
        for cluster in clusters:
            if not cluster:
                routes.append([0, 0])
            else:
                route = [0]
                remaining = set(cluster)
                current = 0
                while remaining:
                    next_node = min(remaining, key=lambda x: distance_matrix[current, x])
                    route.append(next_node)
                    remaining.remove(next_node)
                    current = next_node
                route.append(0)
                route = two_opt(route)
                routes.append(route)
        # Convert to permutation by concatenating route interiors
        perm = []
        for r in routes:
            if len(r) > 2:
                perm.extend(r[1:-1])
        return perm

    # Method 3: Regret construction (from parent1) - convert to permutation
    def regret_perm():
        # Start with empty routes
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = set(customers)
        # Helper for best insertion
        def best_insertion(c):
            best = (float('inf'), -1, -1)
            second = (float('inf'), -1, -1)
            for r_idx, route in enumerate(routes):
                if len(route) < 2:
                    continue
                other_max = 0.0
                for j, d in enumerate(route_dists):
                    if j != r_idx and d > other_max:
                        other_max = d
                for pos in range(1, len(route)):
                    pred = route[pos-1]
                    succ = route[pos]
                    new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                    new_max = max(other_max, new_dist)
                    if new_max < best[0]:
                        best, second = (new_max, r_idx, pos), best
                    elif new_max < second[0]:
                        second = (new_max, r_idx, pos)
            return best[0], best[1], best[2], second[0]
        # Regret construction
        while unassigned:
            bests = []
            for c in unassigned:
                best_new_max, best_route, best_pos, second_new_max = best_insertion(c)
                if best_route == -1:
                    continue
                regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
                bests.append((-regret, c, best_route, best_pos, best_new_max))
            bests.sort(key=lambda x: (x[0], x[1]))
            _, c, best_route, best_pos, new_max = bests[0]
            route = routes[best_route]
            route.insert(best_pos, c)
            route_dists[best_route] = route_dist(route)
            unassigned.remove(c)
        # Convert to permutation
        perm = []
        for r in routes:
            if len(r) > 2:
                perm.extend(r[1:-1])
        return perm

    # Build initial population (each permutation is a list of customers)
    pop_perm = []
    # 1: Nearest neighbor from customer 1
    pop_perm.append(nearest_neighbor_tour(1))
    # 2: Nearest neighbor from customer n-1
    pop_perm.append(nearest_neighbor_tour(n-1))
    # 3: Cluster-first
    pop_perm.append(cluster_first_perm())
    # 4: Regret
    pop_perm.append(regret_perm())
    # 5: Farthest insertion (a different deterministic method)
    # Use cheapest insertion starting from furthest node
    unvisited = set(customers)
    tour = [np.argmax(distance_matrix[0, 1:]) + 1]
    unvisited.remove(tour[0])
    while unvisited:
        best_cost = float('inf')
        best_cust = None
        best_pos = -1
        for cust in unvisited:
            for pos in range(len(tour)+1):
                if pos == 0:
                    cost = distance_matrix[0, cust] + distance_matrix[cust, tour[0]]
                elif pos == len(tour):
                    cost = distance_matrix[tour[-1], cust] + distance_matrix[cust, 0]
                else:
                    cost = distance_matrix[tour[pos-1], cust] + distance_matrix[cust, tour[pos]] - distance_matrix[tour[pos-1], tour[pos]]
                if cost < best_cost:
                    best_cost = cost
                    best_cust = cust
                    best_pos = pos
        tour.insert(best_pos, best_cust)
        unvisited.remove(best_cust)
    pop_perm.append(tour)

    # Ensure all permutations cover exactly all customers (some may miss? They should)
    # Evaluate initial population
    pop_fitness = []
    pop_routes = []
    for perm in pop_perm:
        routes = split_sequence(perm)
        fit = max_dist(routes)
        pop_routes.append(routes)
        pop_fitness.append(fit)
    best_idx = np.argmin(pop_fitness)
    best_routes = [r[:] for r in pop_routes[best_idx]]
    best_fit = pop_fitness[best_idx]
    report_best_vrp(best_routes)

    # -------------------------------
    # GA parameters
    # -------------------------------
    pop_size = 5
    max_gen = 20 * n_cust  # bounded
    for gen in range(max_gen):
        # Sort population by fitness
        order = np.argsort(pop_fitness)
        sorted_pop = [pop_perm[i] for i in order]
        sorted_fitness = [pop_fitness[i] for i in order]
        sorted_routes = [pop_routes[i] for i in order]
        # Select two best parents
        parent1 = sorted_pop[0]
        parent2 = sorted_pop[1]
        # Crossover: Order crossover with fixed point at n//2
        point = n_cust // 2
        def crossover(p1, p2):
            n_c = len(p1)
            child1 = [-1] * n_c
            child2 = [-1] * n_c
            # Copy first segment
            child1[:point] = p1[:point]
            child2[:point] = p2[:point]
            # Fill remaining from other parent
            def fill_child(child, source, start):
                used = set(child[:start])
                pos = start
                for cust in source:
                    if cust not in used:
                        child[pos] = cust
                        pos += 1
                return child
            child1 = fill_child(child1, p2, point)
            child2 = fill_child(child2, p1, point)
            return child1, child2
        child1_perm, child2_perm = crossover(parent1, parent2)
        # Mutation: fixed reversal
        a = n_cust // 3
        b = 2 * n_cust // 3
        def mutate(perm):
            new_perm = perm[:]
            new_perm[a:b+1] = reversed(new_perm[a:b+1])
            return new_perm
        child1_perm = mutate(child1_perm)
        child2_perm = mutate(child2_perm)
        # Evaluate children
        child1_routes = split_sequence(child1_perm)
        child1_fit = max_dist(child1_routes)
        child2_routes = split_sequence(child2_perm)
        child2_fit = max_dist(child2_routes)
        # Replace two worst individuals in population
        # Remove worst two
        pop_perm.pop(order[-1])
        pop_fitness.pop(order[-1])
        pop_routes.pop(order[-1])
        pop_perm.pop(order[-2])
        pop_fitness.pop(order[-2])
        pop_routes.pop(order[-2])
        # Add children
        pop_perm.append(child1_perm)
        pop_fitness.append(child1_fit)
        pop_routes.append(child1_routes)
        pop_perm.append(child2_perm)
        pop_fitness.append(child2_fit)
        pop_routes.append(child2_routes)
        # Update best
        current_best_fit = min(pop_fitness)
        current_best_idx = pop_fitness.index(current_best_fit)
        current_best_routes = pop_routes[current_best_idx]
        if current_best_fit < best_fit - 1e-12:
            best_fit = current_best_fit
            best_routes = [r[:] for r in current_best_routes]
            report_best_vrp(best_routes)
    return best_routes