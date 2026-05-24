import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.best_value = None
        self.best_x = None

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        evals = 0
        popsize = min(budget, max(4, min(5*dim, 20)))
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        fit = np.zeros(popsize)
        for i in range(popsize):
            fit[i] = func(pop[i])
            evals += 1
            if i == 0 or fit[i] < self.best_value:
                self.best_value = fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x

        F = 0.8
        CR = 0.9
        stagnation_counter = 0
        stagnation_limit = max(10 * dim, int(0.2 * budget))

        while evals < budget:
            for i in range(popsize):
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                r1, r2 = candidates[0], candidates[1]
                mutant = pop[i] + F * (self.best_x - pop[i]) + F * (pop[r1] - pop[r2])
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_fit = func(trial)
                evals += 1
                improved_global = False
                if trial_fit <= fit[i]:
                    fit[i] = trial_fit
                    pop[i] = trial
                    if trial_fit < self.best_value:
                        self.best_value = trial_fit
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        improved_global = True
                if improved_global:
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                if evals >= budget:
                    break
            if evals >= budget:
                break
            if stagnation_counter >= stagnation_limit and evals < budget - popsize:
                stagnation_counter = 0
                # sort by fitness, keep best half, replace worst half with uniform random
                sorted_idx = np.argsort(fit)
                keep_num = popsize // 2
                new_pop_indices = sorted_idx[keep_num:]
                for idx in new_pop_indices:
                    if evals >= budget:
                        break
                    pop[idx] = lb + (ub - lb) * rng.rand(dim)
                    fit[idx] = func(pop[idx])
                    evals += 1
                    if fit[idx] < self.best_value:
                        self.best_value = fit[idx]
                        self.best_x = pop[idx].copy()
                        report_best(self.best_value, self.best_x)
        return self.best_value, self.best_x