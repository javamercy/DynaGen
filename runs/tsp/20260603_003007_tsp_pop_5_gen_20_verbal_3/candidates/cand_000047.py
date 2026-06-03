import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    # nearest neighbor construction from city 0
    unvisited = set(range(1, n))
    tour = [0]
    curr = 0
    while unvisited:
        next_city = min(unvisited, key=lambda c: distance_matrix[curr, c])
        tour.append(next_city)
        unvisited.remove(next_city)
        curr = next_city
    best_tour = tour[:]
    best_cost = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    report_best_tour(np.array(best_tour))

    def cost(t):
        total = 0.0
        for i in range(n):
            total += distance_matrix[t[i], t[(i+1)%n]]
        return total

    temp = 100.0
    cooling = 0.995
    max_iter = 20000
    restart_threshold = 1000
    no_improve = 0

    for _ in range(max_iter):
        # random insertion move
        i = random.randint(0, n-1)
        city = tour[i]
        new_tour = tour[:i] + tour[i+1:]
        j = random.randint(0, n-1)
        new_tour.insert(j, city)
        curr_cost = cost(tour)
        new_cost = cost(new_tour)
        delta = new_cost - curr_cost
        if delta < 0 or random.random() < np.exp(-delta / temp):
            tour = new_tour
            if new_cost < best_cost:
                best_cost = new_cost
                best_tour = tour[:]
                report_best_tour(np.array(best_tour))
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        # restart if no improvement for a while
        if no_improve >= restart_threshold:
            start = random.randint(0, n-1)
            unvisited = set(range(n))
            unvisited.discard(start)
            new_tour = [start]
            curr = start
            while unvisited:
                next_city = min(unvisited, key=lambda c: distance_matrix[curr, c])
                new_tour.append(next_city)
                unvisited.remove(next_city)
                curr = next_city
            # perturb with random 2-opt move
            i2 = random.randint(0, n-1)
            k = random.randint(i2+1, n-1)
            new_tour2 = new_tour[:i2] + new_tour[i2:k+1][::-1] + new_tour[k+1:]
            tour = new_tour2
            temp = 100.0
            no_improve = 0

        temp *= cooling

    return np.array(best_tour)