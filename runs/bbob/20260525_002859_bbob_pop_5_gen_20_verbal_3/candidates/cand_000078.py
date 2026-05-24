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
        generation = 0

        while evals < budget:
            generation += 1
            # Shuffle indices for diversity
            indices = list(range(popsize))
            rng.shuffle(indices)
            for i in indices:
                if evals >= budget:
                    break
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                r1, r2 = candidates[0], candidates[1]
                mutant = pop[i] + F * (pop[r1] - pop[r2])
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_fit = func(trial)
                evals += 1
                # Crowding: replace nearest neighbor if better
                dist = np.linalg.norm(pop - trial, axis=1)
                nearest_idx = np.argmin(dist)
                if trial_fit <= fit[nearest_idx]:
                    fit[nearest_idx] = trial_fit
                    pop[nearest_idx] = trial
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
            # Restart if stagnation
            if stagnation_counter >= stagnation_limit and evals < budget - popsize:
                stagnation_counter = 0
                new_pop = [self.best_x.copy()]
                for _ in range(popsize - 1):
                    if rng.rand() < 0.5:
                        sigma = (ub - lb) * 0.2 * (1 + 0.5 * np.random.randn())
                        new_x = self.best_x + rng.randn(dim) * sigma
                    else:
                        new_x = lb + (ub - lb) * rng.rand(dim)
                    new_x = np.clip(new_x, lb, ub)
                    new_pop.append(new_x)
                # Evaluate new individuals (skip best)
                for j in range(1, popsize):
                    if evals >= budget:
                        break
                    x = new_pop[j]
                    f = func(x)
                    evals += 1
                    pop[j] = x
                    fit[j] = f
                    if f < self.best_value:
                        self.best_value = f
                        self.best_x = x.copy()
                        report_best(self.best_value, self.best_x)
                # Reset success memories
                F_success = []
                CR_success = []
                # Local refinement
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