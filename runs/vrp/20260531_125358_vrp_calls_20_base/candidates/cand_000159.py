import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    num_customers = n - 1

    def route_distance(route):
        if len(route) < 2:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def split_permutation(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0] * truck_count
        for cust in perm:
            best_truck = -1
            best_new_max = float('inf')
            best_pos = -1
            best_total = float('inf')
            for t in range(truck_count):
                best_inc = float('inf')
                best_p = -1
                for p in range(1, len(routes[t])):
                    inc = distance_matrix[routes[t][p-1], cust] + distance_matrix[cust, routes[t][p]] - distance_matrix[routes[t][p-1], routes[t][p]]
                    if inc < best_inc:
                        best_inc = inc
                        best_p = p
                new_len = lengths[t] + best_inc
                other_lengths = [lengths[i] for i in range(truck_count) if i != t]
                new_max = max(new_len, max(other_lengths) if other_lengths else 0)
                new_total = new_len + sum(other_lengths)
                if (new_max < best_new_max or
                    (new_max == best_new_max and new_total < best_total) or
                    (new_max == best_new_max and new_total == best_total and t < best_truck)):
                    best_new_max = new_max
                    best_total = new_total
                    best_truck = t
                    best_pos = best_p
            routes[best_truck].insert(best_pos, cust)
            lengths[best_truck] = route_distance(routes[best_truck])
        return routes, lengths

    def two_opt_perm(perm, max_iter=5):
        perm = list(perm)
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(len(perm)-1):
                for j in range(i+1, len(perm)):
                    new_perm = perm[:i] + perm[i:j+1][::-1] + perm[j+1:]
                    old_dist = sum(distance_matrix[perm[k], perm[k+1]] for k in range(len(perm)-1))
                    new_dist = sum(distance_matrix[new_perm[k], new_perm[k+1]] for k in range(len(new_perm)-1))
                    if new_dist < old_dist:
                        perm = new_perm
                        improved = True
        return perm

    # ACO parameters
    num_ants = min(30, max(10, num_customers))
    iterations = min(50, max(10, 2 * num_customers))
    alpha = 1.0
    beta = 2.0
    rho = 0.1
    tau0 = 1.0 / (num_customers * np.mean(distance_matrix) + 1e-10)

    # Initialize pheromone matrix (n x n, but only customer-customer used)
    tau = np.full((n, n), tau0, dtype=float)
    # Heuristic matrix (avoid zero division)
    eta = 1.0 / (distance_matrix + 1e-10)
    # Fix diagonal (no self-loops)
    for i in range(n):
        tau[i, i] = 0.0
        eta[i, i] = 0.0

    best_perm = None
    best_routes = None
    best_max = float('inf')
    best_total = float('inf')

    customer_ids = list(range(1, n))

    for iteration in range(iterations):
        for ant in range(num_ants):
            # Construct giant-tour permutation (customers only)
            unvisited = set(customer_ids)
            perm = []
            # Choose first customer randomly
            first = random.choice(list(unvisited))
            perm.append(first)
            unvisited.remove(first)
            current = first
            while unvisited:
                # Compute selection probabilities
                prob_list = []
                total_prob = 0.0
                for j in unvisited:
                    p = (tau[current, j] ** alpha) * (eta[current, j] ** beta)
                    prob_list.append(p)
                    total_prob += p
                if total_prob == 0:
                    next_cust = random.choice(list(unvisited))
                else:
                    r = random.random() * total_prob
                    cumulative = 0.0
                    for idx, j in enumerate(unvisited):
                        cumulative += prob_list[idx]
                        if cumulative >= r:
                            next_cust = j
                            break
                perm.append(next_cust)
                unvisited.remove(next_cust)
                current = next_cust

            # Optional: 2-opt on the permutation to improve
            perm = two_opt_perm(perm, max_iter=3)

            # Decode permutation into routes
            routes, lengths = split_permutation(perm)
            maxdist = max(lengths)
            totaldist = sum(lengths)
            if maxdist < best_max or (maxdist == best_max and totaldist < best_total):
                best_max = maxdist
                best_total = totaldist
                best_perm = perm[:]
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)

        # Evaporate pheromone
        tau *= (1.0 - rho)
        # Deposit pheromone on best permutation edges
        if best_perm is not None:
            delta = 1.0 / (best_max + 1e-10)
            for i in range(len(best_perm) - 1):
                tau[best_perm[i], best_perm[i+1]] += delta
            # Optionally reinforce edges from depot to first and last to depot
            tau[0, best_perm[0]] += delta
            tau[best_perm[-1], 0] += delta

    return best_routes