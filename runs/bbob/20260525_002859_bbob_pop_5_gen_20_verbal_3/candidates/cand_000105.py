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

        popsize = min(budget // 2, max(4, 5 * dim))
        popsize = max(4, popsize)
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        fit = np.zeros(popsize)

        evals = 0
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
        memory_size = 10
        F_success = []
        CR_success = []
        stagnation_limit = max(10 * dim, int(0.15 * budget))
        stagnation_counter = 0
        generation = 0

        range_vec = ub - lb
        diversity_threshold = 0.2 * np.linalg.norm(range_vec) / np.sqrt(dim)
        diversity_prob = 0.05

        while evals < budget:
            generation += 1
            for i in range(popsize):
                if evals >= budget:
                    break
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                r1, r2, r3 = candidates[0], candidates[1], candidates[2] if len(candidates) > 2 else candidates[0], candidates[1], candidates[2]  # ensure three distinct
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
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
                    if trial_fit < self.best_value:
                        self.best_value = trial_fit
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        stagnation_counter = 0
                        F_success.append(F)
                        CR_success.append(CR)
                    else:
                        stagnation_counter += 1
                else:
                    if rng.rand() < diversity_prob:
                        dist_trial_to_best = np.linalg.norm(trial - self.best_x)
                        if dist_trial_to_best > diversity_threshold:
                            j = rng.randint(popsize)
                            if j != i:
                                pop[j] = trial
                                fit[j] = trial_fit
                    stagnation_counter += 1
                if evals >= budget:
                    break
            if len(F_success) >= memory_size:
                F = np.mean(F_success[-memory_size:])
                CR = np.mean(CR_success[-memory_size:])
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)
            if stagnation_counter >= stagnation_limit and evals < budget - popsize:
                stagnation_counter = 0
                new_pop = [self.best_x.copy()]
                sigma = 0.3 * (ub - lb) * (1 + 0.5 * rng.randn(dim))
                while len(new_pop) < popsize:
                    if rng.rand() < 0.7:
                        x = self.best_x + rng.randn(dim) * sigma
                        x = np.clip(x, lb, ub)
                    else:
                        x = lb + (ub - lb) * rng.rand(dim)
                    new_pop.append(x)
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
                if evals < budget - 1:
                    sigma_local = 0.05 * (ub - lb)
                    for _ in range(min(5, budget - evals)):
                        candidate = self.best_x + rng.randn(dim) * sigma_local
                        candidate = np.clip(candidate, lb, ub)
                        f = func(candidate)
                        evals += 1
                        if f < self.best_value:
                            self.best_value = f
                            self.best_x = candidate.copy()
                            report_best(self.best_value, self.best_x)
                        if evals >= budget:
                            break
                F_success = []
                CR_success = []
                if evals >= budget:
                    break
        return self.best_value, self.best_x