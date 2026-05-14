from __future__ import annotations


TSP_BASELINES = {
    "random_shuffle": r'''import numpy as np
import random

def solve_tsp(distance_matrix, seed, budget):
    n = int(distance_matrix.shape[0])
    if n <= 2:
        return np.arange(n, dtype=int)
    rng = random.Random(int(seed))
    tour = list(range(n))
    rng.shuffle(tour)
    return np.asarray(tour, dtype=int)
''',
    "nearest_neighbor": r'''import numpy as np
import random

def solve_tsp(distance_matrix, seed, budget):
    n = int(distance_matrix.shape[0])
    if n <= 2:
        return np.arange(n, dtype=int)

    random.seed(int(seed))
    start = random.randint(0, n - 1)
    tour = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    current = start

    while unvisited:
        nearest = min(unvisited, key=lambda city: distance_matrix[current, city])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    return np.asarray(tour, dtype=int)
''',
    "two_opt": r'''import numpy as np
import random

def solve_tsp(distance_matrix, seed, budget):
    n = int(distance_matrix.shape[0])
    if n <= 2:
        return np.arange(n, dtype=int)

    random.seed(int(seed))

    def compute_length(tour):
        total = distance_matrix[tour[-1], tour[0]]
        for i in range(len(tour) - 1):
            total += distance_matrix[tour[i], tour[i + 1]]
        return total

    def nearest_neighbor():
        start = random.randint(0, n - 1)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        while unvisited:
            nearest = min(unvisited, key=lambda city: distance_matrix[current, city])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        return tour

    def improve(tour, max_attempts):
        attempts = 0
        improved = True
        while improved and attempts < max_attempts:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if attempts >= max_attempts:
                        return tour
                    a = tour[i]
                    b = tour[(i + 1) % n]
                    c = tour[j]
                    d = tour[(j + 1) % n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-12:
                        tour = tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]
                        improved = True
                        attempts += 1
                        break
                if improved:
                    break
        return tour

    tour = nearest_neighbor()
    max_attempts = min(5000, max(1, int(budget)))
    tour = improve(tour, max_attempts)
    return np.asarray(tour, dtype=int)
''',
    "cheapest_insertion": r'''import numpy as np
import random

def solve_tsp(distance_matrix, seed, budget):
    n = int(distance_matrix.shape[0])
    if n <= 2:
        return np.arange(n, dtype=int)

    random.seed(int(seed))
    remaining = list(range(n))
    start = random.randint(0, n - 1)
    tour = [start]
    remaining.remove(start)

    second = min(remaining, key=lambda city: distance_matrix[start, city])
    tour.append(second)
    remaining.remove(second)

    while remaining:
        best_city = None
        best_pos = None
        best_delta = float('inf')
        for city in remaining:
            for pos in range(len(tour)):
                a = tour[pos]
                b = tour[(pos + 1) % len(tour)]
                delta = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                if delta < best_delta:
                    best_delta = delta
                    best_city = city
                    best_pos = pos + 1
        tour.insert(best_pos, best_city)
        remaining.remove(best_city)

    return np.asarray(tour, dtype=int)
''',
    "random_restart": r'''import numpy as np
import random

def solve_tsp(distance_matrix, seed, budget):
    n = int(distance_matrix.shape[0])
    if n <= 2:
        return np.arange(n, dtype=int)

    random.seed(int(seed))

    def compute_length(tour):
        total = distance_matrix[tour[-1], tour[0]]
        for i in range(len(tour) - 1):
            total += distance_matrix[tour[i], tour[i + 1]]
        return total

    def nearest_neighbor(start):
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        while unvisited:
            nearest = min(unvisited, key=lambda city: distance_matrix[current, city])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        return tour

    def two_opt(tour, max_attempts):
        attempts = 0
        improved = True
        while improved and attempts < max_attempts:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if attempts >= max_attempts:
                        return tour
                    a = tour[i]
                    b = tour[(i + 1) % n]
                    c = tour[j]
                    d = tour[(j + 1) % n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-12:
                        tour = tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]
                        improved = True
                        attempts += 1
                        break
                if improved:
                    break
        return tour

    best_tour = nearest_neighbor(random.randint(0, n - 1))
    best_len = compute_length(best_tour)
    restarts = max(1, min(int(budget), 5))
    for _ in range(restarts):
        start = random.randint(0, n - 1)
        tour = nearest_neighbor(start)
        tour = two_opt(tour, min(500, max(1, int(budget) // restarts)))
        tour_len = compute_length(tour)
        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour

    return np.asarray(best_tour, dtype=int)
''',
}


def get_tsp_baseline_code(name: str) -> str:
    try:
        return TSP_BASELINES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown TSP baseline: {name}") from exc
