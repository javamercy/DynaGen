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
        fit = np.full(popsize, np.inf)

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

        # Initialize per-individual F and CR (jDE style)
        F = 0.5 + 0.3 * rng.rand(popsize)
        CR = 0.5 * np.ones(popsize)
        tau1 = 0.1
        tau2 = 0.1
        FL = 0.1
        FU = 0.9

        # Stagnation detection
        stagnation_limit = max(10 * dim, int(0.15 * budget))
        stagnation_counter = 0
        best_improvement = True

        while evals < budget:
            # Generate offspring
            for i in range(popsize):
                if evals >= budget:
                    break

                # Update F and CR with probability
                if rng.rand() < tau1:
                    F[i] = FL + rng.rand() * (FU - FL)
                if rng.rand() < tau2:
                    CR[i] = rng.rand()

                # Select two distinct random indices != i
                candidates = [j for j in range(popsize) if j != i]
                rng.shuffle(candidates)
                r1, r2 = candidates[0], candidates[1]

                # DE/current-to-best/1/bin
                mutant = pop[i] + F[i] * (self.best_x - pop[i]) + F[i] * (pop[r1] - pop[r2])
                trial = np.where(rng.rand(dim) < CR[i], mutant, pop[i])
                # Ensure at least one component from mutant
                j_rand = rng.randint(dim)
                trial[j_rand] = mutant[j_rand]
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
                        best_improvement = True
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1

                if evals >= budget:
                    break

            # Restart if stagnation and enough budget left
            if stagnation_counter >= stagnation_limit and evals < budget - popsize:
                stagnation_counter = 0
                # Keep best individual
                new_pop = [self.best_x.copy()]
                sorted_idx = np.argsort(fit)
                worst_indices = sorted_idx[1:]  # exclude best
                rng.shuffle(worst_indices)
                for idx in worst_indices:
                    if len(new_pop) >= popsize:
                        break
                    if rng.rand() < 0.5:
                        # perturb best with adaptive step
                        sigma = (ub - lb) * 0.2
                        new_x = self.best_x + rng.randn(dim) * sigma
                        new_x = np.clip(new_x, lb, ub)
                    else:
                        # random uniform
                        new_x = lb + (ub - lb) * rng.rand(dim)
                    new_pop.append(new_x)

                # Evaluate new individuals (except best already evaluated)
                for j, x in enumerate(new_pop[1:], start=1):
                    if evals >= budget:
                        break
                    fit_j = func(x)
                    evals += 1
                    pop[j] = x
                    fit[j] = fit_j
                    if fit_j < self.best_value:
                        self.best_value = fit_j
                        self.best_x = x.copy()
                        report_best(self.best_value, self.best_x)

                # Reset F and CR for reinitialized individuals (keep for best)
                # Randomize F and CR for all except best
                for j in range(1, popsize):
                    if j < len(new_pop):
                        F[j] = 0.5 + 0.3 * rng.rand()
                        CR[j] = 0.5 * rng.rand()

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