import numpy as np
import random

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
        pop_size = min(5 * dim, max(10, budget // 4))
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        # Initialize F and CR for each individual
        F = np.random.uniform(0.1, 1.0, pop_size)
        CR = np.random.uniform(0.0, 1.0, pop_size)
        pop_f = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        fcalls = 0
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
        tau1 = 0.1
        tau2 = 0.1
        while fcalls < budget:
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                # Update F and CR with probabilities tau1, tau2
                if random.random() < tau1:
                    F[i] = np.random.uniform(0.1, 1.0)
                if random.random() < tau2:
                    CR[i] = np.random.uniform(0.0, 1.0)
                # Mutation: DE/rand/1
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2, r3 = random.sample(candidates, 3)
                mutant = pop[r1] + F[i] * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)
                # Crossover: binomial
                trial = pop[i].copy()
                j_rand = random.randint(0, dim-1)
                for j in range(dim):
                    if random.random() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]
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