import numpy as np
from numpy.random import RandomState

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Fallback for small budget
        if budget < 4:
            best_val = np.inf
            best_x = None
            for _ in range(budget):
                x = rng.uniform(lb, ub, size=dim)
                val = func(x)
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        # Population size
        pop_size = max(4, min(4 * dim, budget // 2))
        if pop_size > budget:
            pop_size = budget

        # Initialize population
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        for i in range(pop_size):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_val:
                best_val = fitness[i]
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        if evals >= budget:
            return best_val, best_x

        # DE parameters
        F = 0.8
        CR = 0.9
        stagnation_limit = 20
        gen_without_improvement = 0

        # Main loop
        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Mutation
                candidates = [j for j in range(pop_size) if j != i]
                ids = rng.choice(candidates, 3, replace=False)
                a, b, c = ids
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # Crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                # Selection
                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    fitness[i] = trial_fit
                    pop[i] = trial
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        gen_without_improvement = 0
                    else:
                        gen_without_improvement += 1
                else:
                    gen_without_improvement += 1

                # Check stagnation after each generation (end of for loop)
                if gen_without_improvement >= stagnation_limit:
                    # Restart: keep best, reinitialize others
                    pop_new = rng.uniform(lb, ub, size=(pop_size, dim))
                    pop_new[0] = best_x.copy()
                    fitness_new = np.full(pop_size, np.inf)
                    fitness_new[0] = best_val
                    for j in range(1, pop_size):
                        if evals >= budget:
                            break
                        fitness_new[j] = func(pop_new[j])
                        evals += 1
                        if fitness_new[j] < best_val:
                            best_val = fitness_new[j]
                            best_x = pop_new[j].copy()
                            report_best(best_val, best_x)
                    pop = pop_new.copy()
                    fitness = fitness_new.copy()
                    gen_without_improvement = 0

        return best_val, best_x