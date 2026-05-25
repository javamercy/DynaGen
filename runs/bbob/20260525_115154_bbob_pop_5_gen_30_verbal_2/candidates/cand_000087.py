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

        for i in range(pop_size):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_val:
                best_val = fitness[i]
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # JADE-style parameter adaptation
        mean_F = 0.5
        mean_CR = 0.5
        successful_F = []
        successful_CR = []

        gen_max_restart = max(1, budget // (2 * pop_size))
        gen_no_improve = 0
        prev_best_val = best_val

        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = rng.choice(candidates, size=3, replace=False)
                # Sample F from Cauchy
                F = rng.standard_cauchy() * 0.1 + mean_F
                while F <= 0:
                    F = rng.standard_cauchy() * 0.1 + mean_F
                F = min(F, 1.0)
                # Sample CR from normal
                CR = rng.normal(mean_CR, 0.1)
                CR = np.clip(CR, 0.0, 1.0)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
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
                    successful_F.append(F)
                    successful_CR.append(CR)

            # Update means after each generation
            if successful_F:
                sum_F = sum(successful_F)
                sum_F_sq = sum(f*f for f in successful_F)
                if sum_F > 0:
                    mean_F = sum_F_sq / sum_F
                successful_F = []
            if successful_CR:
                sum_CR = sum(successful_CR)
                sum_CR_sq = sum(cr*cr for cr in successful_CR)
                if sum_CR > 0:
                    mean_CR = sum_CR_sq / sum_CR
                successful_CR = []

            if best_val < prev_best_val:
                gen_no_improve = 0
                prev_best_val = best_val
            else:
                gen_no_improve += 1

            if gen_no_improve >= gen_max_restart and evals < budget:
                new_pop = np.empty((pop_size, dim))
                new_pop[0] = best_x
                # Fill half with Cauchy around best (scale 0.05), half uniform
                for k in range(1, pop_size):
                    if evals >= budget:
                        break
                    if k <= (pop_size - 1) // 2:
                        # Cauchy perturbation
                        pert = rng.standard_cauchy(dim) * 0.05 * (ub - lb)
                        new_point = best_x + pert
                        new_point = np.clip(new_point, lb, ub)
                    else:
                        new_point = rng.uniform(lb, ub, dim)
                    new_pop[k] = new_point
                pop = new_pop
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
                        report_best(best_val, best_x)
                prev_best_val = best_val
                gen_no_improve = 0

        return best_val, best_x