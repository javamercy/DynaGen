import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n)
    # Farthest insertion construction
    tour = _farthest_insertion(distance_matrix)
    best_tour = tour[:]
    best_dist = _tour_distance(best_tour, distance_matrix)
    report_best_tour(np.array(best_tour))
    # Iterative improvement with perturbations
    for _ in range(20):
        tour = _two_opt(tour, distance_matrix)
        dist = _tour_distance(tour, distance_matrix)
        if dist < best_dist:
            best_tour = tour[:]
            best_dist = dist
            report_best_tour(np.array(best_tour))
        tour = _double_bridge(tour)
    # Final 2-opt on best tour
    best_tour = _two_opt(best_tour, distance_matrix)
    final_dist = _tour_distance(best_tour, distance_matrix)
    if final_dist < best_dist:
        report_best_tour(np.array(best_tour))
    return np.array(best_tour)

def _farthest_insertion(dist):
    n = dist.shape[0]
    # Find farthest pair
    max_d = -1
    start, end = 0, 1
    for i in range(n):
        for j in range(i+1, n):
            if dist[i,j] > max_d:
                max_d = dist[i,j]
                start, end = i, j
    tour = [start, end]
    in_tour = {start, end}
    while len(tour) < n:
        farthest = None
        max_min = -1
        for city in range(n):
            if city in in_tour:
                continue
            min_d = min(dist[city, t] for t in tour)
            if min_d > max_min:
                max_min = min_d
                farthest = city
        best_pos = 0
        best_inc = float('inf')
        for pos in range(len(tour)):
            prev = tour[pos]
            nxt = tour[(pos+1) % len(tour)]
            inc = dist[prev,farthest] + dist[farthest,nxt] - dist[prev,nxt]
            if inc < best_inc:
                best_inc = inc
                best_pos = pos+1
        tour.insert(best_pos, farthest)
        in_tour.add(farthest)
    return tour

def _two_opt(tour, dist):
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n-2):
            for j in range(i+2, n):
                a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                if dist[a,c] + dist[b,d] < dist[a,b] + dist[c,d]:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
                    break
            if improved:
                break
    return tour

def _double_bridge(tour):
    n = len(tour)
    if n < 8:
        return tour
    i = random.randint(1, n//3)
    j = random.randint(i+1, n//2)
    k = random.randint(j+1, 2*n//3)
    l = random.randint(k+1, n-1)
    new_tour = tour[:i] + tour[k:l] + tour[j:k] + tour[i:j] + tour[l:]
    return new_tour

def _tour_distance(tour, dist):
    n = len(tour)
    return sum(dist[tour[i], tour[(i+1)%n]] for i in range(n))