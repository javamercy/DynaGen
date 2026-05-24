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

        # Population size: moderate
        popsize = max(4, min(budget // 4, 5 * dim))
        popsize = min(popsize, budget // 2)  # ensure enough budget for generations
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

        # DE strategies: 0 = rand/1/bin, 1 = current-to-best/1/bin
        n_strategies = 2
        strategy_weights = np.ones(n_strategies) / n_strategies
        strategy_success = np.zeros(n_strategies)
        strategy_count = np.ones(n_strategies)  # avoid division by zero

        # Adaptive parameters (not per strategy, but we can have multiple)
        F = 0.8
        CR = 0.9
        memory_F = [F]
        memory_CR = [CR]
        memory_size = 10

        # Stagnation and diversity
        stagnation_counter = 0
        stagnation_limit = max(10 * dim, int(0.2 * budget))
        diversity_threshold = 0.2 * np.linalg.norm(ub - lb) / np.sqrt(dim)
        diversity_prob = 0.1

        # Step history for covariance estimate
        step_history = []
        max_history = 50

        generation = 0
        while evals < budget:
            generation += 1
            # Update strategy weights using exponential smoothing
            # We'll update after each generation
            new_weights = np.zeros(n_strategies)
            for s in range(n_strategies):
                if strategy_count[s] > 0:
                    success_rate = strategy_success[s] / strategy_count[s]
                    new_weights[s] = 0.1 + 0.9 * success_rate  # smoothed
                else:
                    new_weights[s] = 0.1
            strategy_weights = new_weights / new_weights.sum()

            for i in range(popsize):
                if evals >= budget:
                    break
                # Select strategy
                s = rng.choice(n_strategies, p=strategy_weights)
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                r1, r2 = candidates[0], candidates[1]
                if s == 0:  # rand/1/bin
                    mutant = pop[r1] + F * (pop[r2] - pop[r3]) if len(candidates) > 2 else pop[i] + F * (pop[r1] - pop[r2])
                else:  # current-to-best/1/bin
                    mutant = pop[i] + F * (self.best_x - pop[i]) + F * (pop[r1] - pop[r2])
                # Ensure we have r3 for rand/1; if not enough population, fallback
                if s == 0 and len(candidates) > 2:
                    r3 = candidates[2]
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])
                elif s == 0:
                    mutant = pop[i] + F * (pop[r1] - pop[r2])

                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_fit = func(trial)
                evals += 1

                # Selection
                if trial_fit <= fit[i]:
                    fit[i] = trial_fit
                    pop[i] = trial
                    strategy_success[s] += 1
                    strategy_count[s] += 1
                    if trial_fit < self.best_value:
                        self.best_value = trial_fit
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        stagnation_counter = 0
                        step_history.append(trial - self.best_x)  # approximate step
                        if len(step_history) > max_history:
                            step_history.pop(0)
                    else:
                        stagnation_counter += 1
                else:
                    strategy_count[s] += 1
                    stagnation_counter += 1
                    # Diversity preservation
                    if rng.rand() < diversity_prob:
                        dist_trial_to_best = np.linalg.norm(trial - self.best_x)
                        if dist_trial_to_best > diversity_threshold:
                            # Replace a random individual (not best) with trial
                            idx = rng.randint(popsize)
                            if idx != np.argmin(fit):  # avoid replacing best
                                pop[idx] = trial
                                fit[idx] = trial_fit

                if evals >= budget:
                    break

            # Update F and CR from success memory (simple average of recent successes)
            if len(memory_F) >= memory_size:
                F = np.mean(memory_F[-memory_size:])
                CR = np.mean(memory_CR[-memory_size:])
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

            # Restart if stagnation and enough budget left
            if stagnation_counter >= stagnation_limit and evals < budget - popsize:
                stagnation_counter = 0
                # Prepare covariance from step history if available
                if len(step_history) >= 2:
                    steps = np.array(step_history[-min(50, len(step_history)):])
                    cov = np.cov(steps, rowvar=False) + 1e-10 * np.eye(dim)
                else:
                    cov = np.eye(dim) * (0.2 * (ub - lb))**2

                # Reinitialize population
                new_pop = [self.best_x.copy()]
                sigma = 0.3 * (ub - lb) * np.sqrt(np.diag(cov)) / np.linalg.norm(ub - lb)
                while len(new_pop) < popsize:
                    if rng.rand() < 0.7:
                        # Perturb best using covariance
                        perturbation = rng.multivariate_normal(np.zeros(dim), cov)
                        x = self.best_x + 0.5 * perturbation / np.sqrt(dim)
                    else:
                        x = lb + (ub - lb) * rng.rand(dim)
                    x = np.clip(x, lb, ub)
                    new_pop.append(x)
                # Evaluate new individuals (except first which is best already)
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
                # Local refinement on best
                if evals < budget - 2:
                    sigma_local = 0.05 * (ub - lb) / np.sqrt(dim)
                    for _ in range(min(10, budget - evals)):
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
                # Reset success memories
                strategy_success[:] = 0
                strategy_count[:] = 1
                memory_F = []
                memory_CR = []
                step_history = []
                if evals >= budget:
                    break

        return self.best_value, self.best_x