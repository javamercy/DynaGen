import numpy as np
import random
import heapq

def report_best_tour(tour):
    pass # Placeholder as per interface requirements

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    random.seed(seed)
    np.random.seed(seed)

    # 1. Construction: Regret-2 Insertion
    start_node = random.randint(0, n - 1)
    unvisited = set(range(n))
    unvisited.remove(start_node)
    
    remaining = list(unvisited)
    second_node = remaining[np.argmin(distance_matrix[start_node, remaining])]
    unvisited.remove(second_node)
    
    tour = [start_node, second_node]
    
    while unvisited:
        best_regret = -1
        best_node = -1
        best_pos = -1
        
        candidates = list(unvisited)
        if len(candidates) > 60:
            candidates = random.sample(candidates, 60)

        for node in candidates:
            costs = []
            for i in range(len(tour)):
                u = tour[i]
                v = tour[(i + 1) % len(tour)]
                cost = distance_matrix[u, node] + distance_matrix[node, v] - distance_matrix[u, v]
                costs.append(cost)
            
            costs.sort()
            regret = costs[1] - costs[0] if len(costs) > 1 else costs[0]
            
            if regret > best_regret:
                best_regret = regret
                best_node = node
                # Recalculate best_pos for this node
                min_c = float('inf')
                temp_pos = -1
                for i in range(len(tour)):
                    u = tour[i]
                    v = tour[(i + 1) % len(tour)]
                    c = distance_matrix[u, node] + distance_matrix[node, v] - distance_matrix[u, v]
                    if c < min_c:
                        min_c = c
                        temp_pos = i
                best_pos = temp_pos
        
        if best_node == -1:
            best_node = random.choice(list(unvisited))
            best_pos = random.randint(0, len(tour) - 1)
            
        tour.insert(best_pos + 1, best_node)
        unvisited.remove(best_node)

    current_tour = np.array(tour)
    report_best_tour(current_tour)

    # 2. Local Search (Budget-bounded 2-opt)
    def get_tour_dist(t):
        d = 0
        for i in range(n):
            d += distance_matrix[t[i], t[(i+1)%n]]
        return d

    best_dist = get_tour_dist(current_tour)
    
    iters = 0
    improved = True
    while improved and iters < budget:
        improved = False
        # Systematic scan
        for i in range(n - 1):
            if iters >= budget: break
            for j in range(i + 2, n):
                if iters >= budget: break
                
                u, v = current_tour[i], current_tour[i+1]
                w, z = current_tour[j], current_tour[(j+1)%n]
                
                delta = (distance_matrix[u, w] + distance_matrix[v, z]) - \
                        (distance_matrix[u, v] + distance_matrix[w, z])
                
                if delta < -1e-9:
                    current_tour[i+1:j+1] = current_tour[i+1:j+1][::-1]
                    best_dist += delta
                    improved = True
                    report_best_tour(current_tour)
                    # Continue scanning from this point rather than full restart
                    # to better utilize the budget
                    break
            iters += 1
        
        # Random restart to escape local optima if budget permits
        if not improved and iters < budget:
            # Try a few random swaps to shake the tour
            for _ in range(min(10, budget - iters)):
                idx1, idx2 = random.sample(range(n), 2)
                current_tour[idx1], current_tour[idx2] = current_tour[idx2], current_tour[idx1]
                # Only keep if it's not disastrously bad, or just use it to jump
                # For simplicity, we just shake and set improved=True to re-run 2-opt
                improved = True
                iters += 1
                break

    return current_tour