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
        fit = np.empty(popsize)
        for i in range(popsize):
            fit[i] = func(pop[i])
            evals += 1
            if evals == 1 or fit[i] < self.best_value:
                self.best_value = fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x

        stag_limit = max(10 * dim, int(0.2 * budget))
        stag_counter = 0
        memory_size = max(10, min(100, 10*dim))
        mF = 0.5
        mCR = 0.9
        success_F = np.empty(memory_size)
        success_CR = np.empty(memory_size)
        success_count = 0

        while evals < budget:
            F_values = np.empty(popsize)
            CR_values = np.empty(popsize)
            for i in range(popsize):
                F = rng.standard_cauchy() * 0.1 + mF
                F = np.clip(F, 0.1, 0.9)
                F_values[i] = F
                CR = rng.standard_normal() * 0.1 + mCR
                CR = np.clip(CR, 0.0, 1.0)
                CR_values[i] = CR

            for i in range(popsize):
                if evals >= budget:
                    break
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                r1, r2 = candidates[0], candidates[1]
                mutant = pop[i] + F_values[i] * (self.best_x - pop[i]) + F_values[i] * (pop[r1] - pop[r2])
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR_values[i] or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_fit = func(trial)
                evals += 1
                if trial_fit <= fit[i]:
                    if success_count < memory_size:
                        success_F[success_count] = F_values[i]
                        success_CR[success_count] = CR_values[i]
                        success_count += 1
                    else:
                        idx = rng.randint(memory_size)
                        success_F[idx] = F_values[i]
                        success_CR[idx] = CR_values[i]
                    fit[i] = trial_fit
                    pop[i] = trial
                    if trial_fit < self.best_value:
                        self.best_value = trial_fit
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        stag_counter = 0
                    else:
                        stag_counter += 1
                else:
                    stag_counter += 1
                if evals >= budget:
                    break
            if evals >= budget:
                break

            if success_count > 0:
                mF = np.mean(success_F[:success_count])
                mCR = np.mean(success_CR[:success_count])

            if stag_counter >= stag_limit and evals < budget - popsize:
                stag_counter = 0
                new_pop = [self.best_x.copy()]
                sorted_idx = np.argsort(fit)
                worst_indices = sorted_idx[1:]
                rng.shuffle(worst_indices)
                for idx in worst_indices:
                    if len(new_pop) >= popsize:
                        break
                    if rng.rand() < 0.5:
                        sigma = (ub - lb) * 0.2 * (0.5 + rng.rand())
                        new_x = self.best_x + rng.randn(dim) * sigma
                        new_x = np.clip(new_x, lb, ub)
                    else:
                        new_x = lb + (ub - lb) * rng.rand(dim)
                    new_pop.append(new_x)
                while len(new_pop) < popsize:
                    new_x = lb + (ub - lb) * rng.rand(dim)
                    new_pop.append(new_x)
                for j, x in enumerate(new_pop[1:], start=1):
                    if evals >= budget:
                        break
                    fit_val = func(x)
                    evals += 1
                    pop[j] = x
                    fit[j] = fit_val
                    if fit_val < self.best_value:
                        self.best_value = fit_val
                        self.best_x = x.copy()
                        report_best(self.best_value, self.best_x)
                if evals >= budget:
                    break
                fit[0] = self.best_value
        return self.best_value, self.best_x