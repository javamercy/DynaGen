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
        popsize = min(budget // 2, max(10, 4 * dim))
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
        diversity_threshold = 0.05 * np.mean(ub - lb)  # threshold for restart

        while evals < budget:
            generation += 1
            # Generate offspring
            for i in range(popsize):
                if evals >= budget:
                    break
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                r1, r2 = candidates[0], candidates[1]
                # Mutation: current-to-best/1
                mutant = pop[i] + F * (self.best_x - pop[i]) + F * (pop[r1] - pop[r2])
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_fit = func(trial)
                evals += 1

                # Crowding selection: replace nearest neighbor if trial is better
                # Find nearest neighbor (excluding current i)
                distances = np.linalg.norm(pop - trial, axis=1)
                distances[i] = np.inf  # exclude self
                nearest_idx = np.argmin(distances)
                if trial_fit < fit[nearest_idx]:
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

            # Update adaptive parameters based on success memory
            if len(F_success) >= memory_size:
                F = np.mean(F_success[-memory_size:])
                CR = np.mean(CR_success[-memory_size:])
                F = np.clip(F, 0.2, 1.5)
                CR = np.clip(CR, 0.0, 1.0)

            # Check diversity for restart
            if evals < budget - popsize:
                pop_std = np.mean(np.std(pop, axis=0))
                if pop_std < diversity_threshold or (stagnation_counter >= stagnation_limit):
                    stagnation_counter = 0
                    # Keep best individual
                    new_pop = [self.best_x.copy()]
                    # Reinitialize rest uniformly (except best)
                    while len(new_pop) < popsize:
                        if rng.rand() < 0.2:  # 20% perturbed best
                            sigma = (ub - lb) * 0.2 * (1 + 0.5 * rng.randn())
                            new_x = self.best_x + rng.randn(dim) * sigma
                            new_x = np.clip(new_x, lb, ub)
                        else:
                            new_x = lb + (ub - lb) * rng.rand(dim)
                        new_pop.append(new_x)
                    # Evaluate new individuals (skip best)
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
                    # Local refinement on best (3 steps)
                    if evals < budget - 1:
                        sigma = (ub - lb) * 0.05
                        for _ in range(min(3, budget - evals)):
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