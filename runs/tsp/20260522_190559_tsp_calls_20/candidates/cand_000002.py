import numpy as np
import random
import heapq

def report_best_tour(tour):
    # This function is expected by the environment to track the best tour found
    pass

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n)

    random.seed(seed)
    np.random.seed(seed)

    def get_tour_dist(tour):
        d = 0
        for i in range(n):
            d += distance_matrix[tour[i], tour[(i + 1) % n]]
        return d

    # 1. Construction: Regret-Insertion
    # Start with a random seed node
    start_node = random.randint(0, n - 1)
    unvisited = set(range(n))
    unvisited.remove(start_node)
    tour = [start_node]

    while unvisited:
        best_regret = -1
        best_node = -1
        best_pos = -1

        # To keep construction fast for large N, we sample a subset of candidates if N is huge
        candidates = list(unvisited)
        if len(candidates) > 50:
            candidates = random.sample(candidates, 50)

        for node in candidates:
            # Calculate costs of inserting 'node' between every pair in current tour
            costs = []
            for i in range(len(tour)):
                u, v = tour[i], tour[(i + 1) % len(tour)]
                cost = distance_matrix[u, node] + distance_matrix[node, v] - distance_matrix[u, v]
                costs.append(cost)
            
            sorted_costs = sorted(costs)
            # Regret is the difference between the best and second-best insertion point
            regret = sorted_costs[1] - sorted_costs[0] if len(sorted_costs) > 1 else sorted_costs[0]
            
            if regret > best_regret:
                best_regret = regret
                best_node = node
                best_pos = np.argmin(costs)

        if best_node == -1: # Fallback
            best_node = unvisited.pop()
            tour.append(best_node)
        else:
            unvisited.remove(best_node)
            # Insert into the tour list
            # Note: tour is a list, we insert at the index that minimizes cost
            # Since the cost calculation included the wrap-around (last to first), 
            # if best_pos is the last index, it means insert between tour[-1] and tour[0].
            if best_pos == len(tour) - 1:
                # This is actually handled by the logic, but let's be precise
                # If we insert at the end, it's between tour[-1] and tour[0]
                # The list.insert method handles indices
                tour.insert(best_pos + 1, best_node)
            else:
                tour.insert(best_pos + 1, best_node)
            # Correction for the wrap-around case: if the best pos was the very last edge
            # the insert might have shifted. Let's use a simpler insertion logic.
            # Recalculate based on the actual list length

    # Re-verify tour length and uniqueness
    tour = list(dict.fromkeys(tour)) # Deduplicate while preserving order
    if len(tour) < n:
        for i in range(n):
            if i not in tour: tour.append(i)

    current_tour = np.array(tour)
    current_dist = get_tour_dist(current_tour)
    report_best_tour(current_tour)

    # 2. Local Search: Budgeted 2-opt
    # Use a simple counter for budget
    iterations = 0
    improved = True
    while improved and iterations < budget:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                iterations += 1
                if iterations >= budget: break
                
                # Current edges: (i, i+1) and (j, j+1)
                # New edges: (i, j) and (i+1, j+1)
                u, v = current_tour[i], current_tour[i+1]
                w, z = current_tour[j], current_tour[(j+1)%n]
                
                gain = (distance_matrix[u, v] + distance_matrix[w, z]) - \n                      (distance_matrix[u, w] + distance_matrix[v, z])
                
                if gain > 0:
                    current_tour[i+1:j+1] = current_tour[i+1:j+1][::-1]
                    current_dist -= gain
                    report_best_tour(current_tour)
                    improved = True
            if iterations >= budget: break

    return current_tour