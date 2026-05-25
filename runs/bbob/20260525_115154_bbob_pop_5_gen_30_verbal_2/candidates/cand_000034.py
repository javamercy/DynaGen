import numpy as np
from numpy.random import RandomState

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        range_ = ub - lb

        pop_size = max(4 * dim, 3)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)
        if pop_size < 3:
            pop_size = 3

        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        # Initialize F and CR for each individual
        F = rng.uniform(0.1, 0.9, size=pop_size)
        CR = rng.uniform(0.1, 0.9, size=pop_size)
        tau1 = 0.1
        tau2 = 0.1
        Fl = 0.1
        Fu = 0.9

        for i in range(pop_size):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_val:
                best_val = fitness[i]
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        stagnation_generations = 0
        max_stagnation = 5 * dim

        generation = 0
        while evals < budget:
            generation += 1
            improved = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Adaptive update of F and CR
                if rng.rand() < tau1:
                    F[i] = Fl + rng.rand() * Fu
                if rng.rand() < tau2:
                    CR[i] = rng.rand()
                # Ensure CR in [0,1]
                CR[i] = np.clip(CR[i], 0, 1)

                # Select three distinct indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                ids = rng.choice(candidates, size=3, replace=False)
                a, b, c = ids
                mutant = pop[a] + F[i] * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)

                # Binomial crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]

                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    fitness[i] = trial_fit
                    pop[i] = trial
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        improved = True

            if not improved:
                stagnation_generations += 1
            else:
                stagnation_generations = 0

            if stagnation_generations >= max_stagnation and evals < budget:
                order = np.argsort(fitness)
                pop = pop[order]
                fitness = fitness[order]
                F = F[order]
                CR = CR[order]
                keep = max(1, int(0.3 * pop_size))
                new_pop_size = pop_size - keep
                if new_pop_size > 0:
                    new_pop = rng.uniform(lb, ub, size=(new_pop_size, dim))
                    new_F = rng.uniform(0.1, 0.9, size=new_pop_size)
                    new_CR = rng.uniform(0.1, 0.9, size=new_pop_size)
                    for i in range(new_pop_size):
                        idx = keep + i
                        if evals >= budget:
                            break
                        new_fit = func(new_pop[i])
                        evals += 1
                        fitness[idx] = new_fit
                        pop[idx] = new_pop[i]
                        F[idx] = new_F[i]
                        CR[idx] = new_CR[i]
                        if new_fit < best_val:
                            best_val = new_fit
                            best_x = new_pop[i].copy()
                            report_best(best_val, best_x)
                stagnation_generations = 0

        return best_val, best_x