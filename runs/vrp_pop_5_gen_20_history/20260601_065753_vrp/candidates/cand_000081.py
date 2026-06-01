import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    def compute_route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def calculate_insertion_cost(route, cust, pos):
        prev = route[pos-1]
        nxt = route[pos]
        return distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
    
    def best_insertion(route, cust, current_length):
        best_delta = float('inf')
        best_pos = -1
        for p in range(1, len(route)):
            delta = calculate_insertion_cost(route, cust, p)
            if delta < best_delta or (delta == best_delta and p < best_pos):
                best_delta = delta
                best_pos = p
        return best_pos, best_delta
    
    def construct_solution():
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0.0] * truck_count
        unvisited = set(customers)
        while unvisited:
            # Compute selection probabilities
            pairs = []
            scores = []
            for r in range(truck_count):
                if lengths[r] == float('inf'):
                    continue
                last = routes[r][-1] if len(routes[r]) > 1 else 0
                for c in list(unvisited):
                    pos, delta = best_insertion(routes[r], c, lengths[r])
                    new_len = lengths[r] + delta
                    heuristic = 1.0 / (distance_matrix[last, c] + 1e-10)
                    pheromone = tau[last, c]
                    # Encourage shorter routes: inverse of (1+lengths[r])
                    balance = 1.0 / (1.0 + lengths[r])
                    score = (pheromone ** alpha) * (heuristic ** beta) * balance
                    pairs.append((r, c, pos))
                    scores.append(score)
            if not pairs:
                break
            total = sum(scores)
            if total == 0:
                # fallback to random
                r = random.randrange(truck_count)
                c = random.choice(list(unvisited))
                pos, delta = best_insertion(routes[r], c, lengths[r])
                pairs = [(r, c, pos)]
            else:
                probs = [s / total for s in scores]
                idx = random.choices(range(len(pairs)), weights=probs, k=1)[0]
                pairs = [pairs[idx]]
            r, c, pos = pairs[0]
            routes[r].insert(pos, c)
            lengths[r] = compute_route_length(routes[r])
            unvisited.remove(c)
        return routes, lengths
    
    def local_search(routes, lengths):
        improved = True
        iterations = 0
        max_iter_local = 10 * (n + truck_count)
        while improved and iterations < max_iter_local:
            improved = False
            iterations += 1
            for r in range(truck_count):
                route = routes[r]
                if len(route) <= 3:
                    continue
                best_delta = 0.0
                best_ij = None
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_len = compute_route_length(new_route)
                        delta = new_len - lengths[r]
                        if delta < best_delta:
                            new_max = max(new_len, max(lengths[:r] + lengths[r+1:], default=0))
                            if new_max < max(lengths):
                                best_delta = delta
                                best_ij = (i, j)
                if best_ij is not None:
                    i, j = best_ij
                    routes[r] = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    lengths[r] = compute_route_length(routes[r])
                    improved = True
        return routes, lengths
    
    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = max(compute_route_length(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]
    
    # Parameters
    num_ants = min(30, n)
    max_iter = 5 * n
    alpha = 1.0
    beta = 2.0
    rho = 0.1
    tau0 = 1.0 / (n * np.mean(distance_matrix) if np.mean(distance_matrix) > 0 else 1.0)
    
    # Initialize pheromone matrix
    tau = np.full((n, n), tau0)
    np.fill_diagonal(tau, 0.0)
    
    best_max = float('inf')
    best_routes = None
    
    for iteration in range(max_iter):
        for ant in range(num_ants):
            routes, lengths = construct_solution()
            routes, lengths = local_search(routes, lengths)
            report_best_vrp(routes)
        # Evaporation
        tau *= (1.0 - rho)
        # Deposit on global best
        if best_routes is not None:
            deposit = 1.0 / best_max if best_max > 0 else 1.0
            for route in best_routes:
                for i in range(len(route)-1):
                    u, v = route[i], route[i+1]
                    tau[u, v] += deposit
                    tau[v, u] += deposit  # symmetric
    return best_routes