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

        # Population size
        pop_size = max(4 * dim, 3)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)
        if pop_size < 3:
            pop_size = 3

        # Initial population
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

        # Parameters
        mu_CR = 0.5  # mean for CR adaptation
        c = 0.1  # learning rate for mu_CR
        # Stagnation detection
        gen_max_restart = max(1, budget // (2 * pop_size))
        prev_best_val = best_val
        gen_no_improve = 0

        while evals < budget:
            # Successful CR values in this generation
            succ_CR = []

            for i in range(pop_size):
                if evals >= budget:
                    break

                # Select three distinct random indices
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c_idx = rng.choice(candidates, size=3, replace=False)

                # Mutation
                F = rng.uniform(0.5, 1.0)
                mutant = pop[a] + F * (pop[b] - pop[c_idx])
                mutant = np.clip(mutant, lb, ub)

                # Crossover with adaptive CR
                CR_i = np.clip(rng.normal(mu_CR, 0.1), 0.0, 1.0)
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR_i or j == j_rand:
                        trial[j] = mutant[j]

                # Evaluate
                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    fitness[i] = trial_fit
                    pop[i] = trial
                    succ_CR.append(CR_i)
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()

            # Update mu_CR
            if len(succ_CR) > 0:
                mu_CR = (1 - c) * mu_CR + c * np.mean(succ_CR)

            # Stagnation check
            if best_val < prev_best_val:
                gen_no_improve = 0
                prev_best_val = best_val
            else:
                gen_no_improve += 1

            if gen_no_improve >= gen_max_restart and evals < budget:
                # Restart: keep best, diversify rest
                new_pop = np.empty((pop_size, dim))
                new_pop[0] = best_x
                # For each remaining individual, with probability 0.5 Cauchy around best, else uniform
                scale_factor = 0.1 * (ub - lb)
                for i in range(1, pop_size):
                    if rng.rand() < 0.5:
                        # Cauchy perturbation of best
                        cauchy_noise = rng.standard_cauchy(size=dim)
                        new_x = best_x + scale_factor * cauchy_noise
                    else:
                        new_x = rng.uniform(lb, ub, size=dim)
                    new_pop[i] = np.clip(new_x, lb, ub)
                pop = new_pop
                # Evaluate new individuals (except best)
                fitness = np.full(pop_size, np.inf)
                fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    fitness[i] = func(pop[i])
                    evals += 1
                    if fitness[i] < best_val:
                        best_val = fitness[i]
                        best_x = pop[i].copy()
                prev_best_val = best_val
                gen_no_improve = 0

        return best_val, best_x