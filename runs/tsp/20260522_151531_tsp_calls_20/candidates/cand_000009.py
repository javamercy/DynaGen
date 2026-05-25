import numpy as np
import random
import math

def report_best_tour(tour):
    pass

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.arange(n)
    if n == 2:
        return np.array([0, 1])

    random.seed(seed)
    np.random.seed(seed)

    # 1. Construction: Regret-2 Insertion (from cand_000005)
    start_node = random.randint(0, n - 1)
    unvisited = set(range(n))
    unvisited.remove(start_node)
    
    remaining = list(unvisited)
    second_node = remaining[np.argmin(distance_matrix[start_node, remaining])]
    unvisited.remove(second_node)
    
    tour_list = [start_node, second_node]
    
    while unvisited:
        best_regret = -1
        best_node = -1
        best_pos = -1
        
        candidates = list(unvisited)
        if len(candidates) > 60:
            candidates = random.sample(candidates, 60)

        for node in candidates:
            costs = []
            for i in range(len(tour_list)):
                u = tour_list[i]
                v = tour_list[(i + 1) % len(tour_list)]
                cost = distance_matrix[u, node] + distance_matrix[node, v] - distance_matrix[u, v]
                costs.append(cost)
            
            costs.sort()
            regret = costs[1] - costs[0] if len(costs) > 1 else costs[0]
            
            if regret > best_regret:
                best_regret = regret
                best_node = node
                min_c = float('inf')
                for i in range(len(tour_list)):
                    u = tour_list[i]
                    v = tour_list[(i + 1) % len(tour_list)]
                    c = distance_matrix[u, node] + distance_matrix[node, v] - distance_matrix[u, v]
                    if c < min_c:
                        min_c = c
                        best_pos = i
        
        if best_node == -1:
            best_node = random.choice(list(unvisited))
            best_pos = random.randint(0, len(tour_list) - 1)
            
        tour_list.insert(best_pos + 1, best_node)
        unvisited.remove(best_node)

    current_tour = np.array(tour_list)
    
    def get_tour_dist(t):
        d = 0
        for i in range(n):
            d += distance_matrix[t[i], t[(i+1)%n]]
        return d

    current_dist = get_tour_dist(current_tour)
    best_tour = np.copy(current_tour)
    best_dist = current_dist
    report_best_tour(best_tour)

    # 2. Candidate List for targeted search (from cand_000001)
    k_neighbors = 20 if n >= 80 else n - 1
    candidates_list = []
    for i in range(n):
        idx = np.argsort(distance_matrix[i])[:k_neighbors + 1]
        candidates_list.append(set(idx))

    # 3. Local Search: SA-hybrid 2-opt (from cand_000004)
    t_start = 10.0
    t_end = 0.01
    cooling_rate = math.pow(t_end / t_start, 1.0 / budget) if budget > 0 else 1.0
    temp = t_start

    iters = 0
    while iters < budget:
        iters += 1
        # Pick a random edge (i, i+1)
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j: continue
        if i > j: i, j = j, i

        # Nodes for 2-opt: reverse [i+1, j]
        u, v = current_tour[i], current_tour[(i + 1) % n]
        w, z = current_tour[j], current_tour[(j + 1) % n]

        # Only attempt if nodes are promising (candidate check)
        if n < 80 or (u in candidates_list[w] or v in candidates_list[z]):
            delta = (distance_matrix[u, w] + distance_matrix[v, z]) - \
                    (distance_matrix[u, v] + distance_matrix[w, z])

            if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / temp)):
                # Apply 2-opt reverse
                current_tour[i+1 : j+1] = current_tour[i+1 : j+1][::-1]
                current_dist += delta
                
                if current_dist < best_dist:
                    best_dist = current_dist
                    best_tour = np.copy(current_tour)
                    report_best_tour(best_tour)
        
        temp *= cooling_rate

    return best_tour