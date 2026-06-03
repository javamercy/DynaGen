import numpy as np
from numpy.random import RandomState

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub
        range_ = ub - lb

        pop_size = max(4 * dim, 10)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)

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

        # Initialize adaptation memories
        mu_F = 0.5
        mu_CR = 0.5
        archive = []  # not used for simplicity

        stagnation = 0
        max_stagnation = 5 * dim

        while evals < budget:
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            improved = False

            for i in range(pop_size):
                if evals >= budget:
                    break

                # Generate F and CR
                F = 0.5 + 0.3 * rng.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = mu_CR + 0.1 * rng.randn()
                CR = np.clip(CR, 0, 1)

                # Select base vector and two others (differential)
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(candidates, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    new_fitness[i] = trial_fit
                    new_pop[i] = trial
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        improved = True
            
            # Update adaptation memories based on successful trials (simplified)
            if improved:
                # Update mu_F and mu_CR with moving average
                mu_F = 0.9 * mu_F + 0.1 * F
                mu_CR = 0.9 * mu_CR + 0.1 * CR
                stagnation = 0
            else:
                stagnation += 1

            pop = new_pop
            fitness = new_fitness

            # Restart if stagnation
            if stagnation >= max_stagnation and evals < budget:
                order = np.argsort(fitness)
                pop = pop[order]
                fitness = fitness[order]
                keep = max(1, int(0.25 * pop_size))
                new_size = pop_size - keep
                # Reinitialize rest uniformly with small perturbation around best
                new_pop = rng.uniform(lb, ub, size=(new_size, dim))
                # perturb 
                for j in range(new_size):
                    new_pop[j] += 0.1 * range_ * rng.randn(dim)
                new_pop = np.clip(new_pop, lb, ub)
                pop[keep:] = new_pop
                for i in range(keep, pop_size):
                    if evals >= budget:
                        break
                    fitness[i] = func(pop[i])
                    evals += 1
                    if fitness[i] < best_val:
                        best_val = fitness[i]
                        best_x = pop[i].copy()
                        report_best(best_val, best_x)
                stagnation = 0

        return best_val, best_x