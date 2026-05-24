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
        popsize = min(budget, max(4, min(5 * dim, 20)))
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        fit = np.zeros(popsize)
        for i in range(popsize):
            fit[i] = func(pop[i])
            evals += 1
            if evals == 1 or fit[i] < self.best_value:
                self.best_value = fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x

        F = 0.8
        CR = 0.9
        stagnation_limit = max(10 * dim, int(0.2 * budget))
        stagnation_counter = 0
        best_in_run = self.best_value
        success_window = 5 * popsize
        success_count = 0
        gen_count = 0

        while evals < budget:
            gen_success = 0
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
                if trial_fit <= fit[i]:
                    fit[i] = trial_fit
                    pop[i] = trial
                    gen_success += 1
                    if trial_fit < self.best_value:
                        self.best_value = trial_fit
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1
                if evals >= budget:
                    break
            if evals >= budget:
                break
            gen_count += 1
            success_count += gen_success
            # Adjust F and CR based on sliding window of recent success rate
            if gen_count % 5 == 0:
                success_rate = success_count / (5 * popsize)  # since window is 5 generations
                if success_rate < 0.3:
                    F = max(0.5, F + 0.1)
                    CR = max(0.5, CR - 0.1)
                elif success_rate > 0.7:
                    F = min(1.0, F - 0.1)
                    CR = min(1.0, CR + 0.1)
                else:
                    F = 0.8
                    CR = 0.9
                F = np.clip(F, 0.2, 1.0)
                CR = np.clip(CR, 0.0, 1.0)
                success_count = 0
            # Restart if stagnation and enough budget left
            if stagnation_counter >= stagnation_limit and evals < budget - popsize:
                stagnation_counter = 0
                # Reinitialize all except best individual
                new_pop = [self.best_x.copy()]
                for _ in range(popsize - 1):
                    if rng.rand() < 0.5:
                        new_x = lb + (ub - lb) * rng.rand(dim)
                    else:
                        sigma = (ub - lb) * 0.2
                        new_x = self.best_x + rng.randn(dim) * sigma
                        new_x = np.clip(new_x, lb, ub)
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
                # Reset F and CR after restart
                F = 0.8
                CR = 0.9
                success_count = 0
        return self.best_value, self.best_x