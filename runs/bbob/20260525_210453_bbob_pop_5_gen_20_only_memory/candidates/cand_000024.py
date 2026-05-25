import numpy as np
import random
import math

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Determine population size, ensure at least 3 for DE mutation
        pop_size = min(5 * dim, max(10, budget // 4))
        if pop_size < 3:
            pop_size = 3
        if pop_size > budget:
            pop_size = budget

        # If budget <= 3, fall back to pure random search (evaluate all points)
        if budget <= 3:
            pop_size = budget
            points = np.random.uniform(lb, ub, (pop_size, dim))
            best_f = np.inf
            best_x = None
            for i in range(pop_size):
                x = np.clip(points[i], lb, ub)
                val = func(x)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                    report_best(best_f, best_x)
            return best_f, best_x

        # Initialization
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        pop_f = np.ones(pop_size) * np.inf
        best_f = np.inf
        best_x = None
        fcalls = 0

        # Evaluate initial population
        for i in range(pop_size):
            if fcalls >= budget:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            fcalls += 1
            pop_f[i] = val
            if val < best_f:
                best_f = val
                best_x = x.copy()
                report_best(best_f, best_x)

        # Main DE loop
        F = 0.8
        CR = 0.9
        while fcalls < budget:
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                # Mutation: select three distinct random indices not equal to i
                indices = list(range(pop_size))
                indices.remove(i)
                if len(indices) < 3:
                    # Should not happen since pop_size >= 3
                    continue
                r0, r1, r2 = random.sample(indices, 3)
                mutant = pop[r0] + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # Crossover
                trial = pop[i].copy()
                j_rand = random.randint(0, dim-1)
                for j in range(dim):
                    if random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                # Evaluate
                val = func(trial)
                fcalls += 1
                if val < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = val
                    if val < best_f:
                        best_f = val
                        best_x = trial.copy()
                        report_best(best_f, best_x)
        return best_f, best_x