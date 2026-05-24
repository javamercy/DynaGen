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

        # Population size
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

        # Adaptive parameters
        F = 0.8
        CR = 0.9
        memory_size = 10
        F_success = []
        CR_success = []
        stagnation_limit = max(10 * dim, int(0.15 * budget))
        stagnation_counter = 0
        diversity_threshold = 0.01 * np.mean(ub - lb)
        generation = 0

        while evals < budget:
            generation += 1
            # Generate offspring
            for i in range(popsize):
                if evals >= budget:
                    break
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                r1, r2, r3 = candidates[0], candidates[1], candidates[2]
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_fit = func(trial)
                evals += 1
                # Crowding replacement: accept if trial better and not too close to best
                if trial_fit < fit[i]:
                    dist_to_best = np.linalg.norm(trial - self.best_x) / np.sqrt(dim)
                    range_norm = np.mean(ub - lb)
                    crowding_dist = 0.1 * range_norm
                    accept = False
                    if trial_fit < self.best_value:
                        accept = True
                    elif dist_to_best > crowding_dist:
                        accept = True
                    else:
                        accept = rng.rand() < 0.5
                    if accept:
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
                    stagnation_counter += 1
                if evals >= budget:
                    break

            # Update adaptive parameters
            if len(F_success) >= memory_size:
                F = np.mean(F_success[-memory_size:])
                CR = np.mean(CR_success[-memory_size:])
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

            # Check stagnation: diversity or no improvement
            diversity = np.mean([np.linalg.norm(pop[j] - self.best_x) for j in range(popsize)]) / np.sqrt(dim)
            if stagnation_counter >= stagnation_limit or (diversity < diversity_threshold and evals < budget - popsize):
                stagnation_counter = 0
                # Restart
                new_pop = [self.best_x.copy()]
                sorted_idx = np.argsort(fit)
                worst_indices = sorted_idx[1:]
                rng.shuffle(worst_indices)
                for idx in worst_indices:
                    if len(new_pop) >= popsize:
                        break
                    if rng.rand() < 0.5:
                        sigma = (ub - lb) * 0.2 * (1 + 0.5 * rng.randn())
                        new_x = self.best_x + rng.randn(dim) * sigma
                        new_x = np.clip(new_x, lb, ub)
                    else:
                        new_x = lb + (ub - lb) * rng.rand(dim)
                    new_pop.append(new_x)
                # Evaluate new individuals (except best already evaluated)
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
                # Reset success memories
                F_success = []
                CR_success = []
                # Local refinement on best
                if evals < budget - 1:
                    sigma = (ub - lb) * 0.05
                    for _ in range(min(5, budget - evals)):
                        candidate = self.best_x + rng.randn(dim) * sigma
                        candidate = np.clip(candidate, lb, ub)
                        f = func(candidate)
                        evals += 1
                        if f < self.best_value:
                            self.best_value = f
                            self.best_x = candidate.copy()
                            report_best(self.best_value, self.best_x)
                        if evals >= budget:
                            break
                if evals >= budget:
                    break

        return self.best_value, self.best_x