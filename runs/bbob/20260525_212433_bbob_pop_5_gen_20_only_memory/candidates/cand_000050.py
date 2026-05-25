import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        budget = self.budget
        rng = self.rng

        # Small budget: pure random search
        if budget < 3:
            while evals < budget:
                x = lb + (ub - lb) * rng.rand(self.dim)
                val = func(x)
                evals += 1
                if val < self.best_val:
                    self.best_val = val
                    self.best_x = x.copy()
                    report_best(self.best_val, self.best_x)
            return self.best_val, self.best_x

        # Normal: Differential Evolution with exploration and restarts
        pop_size = min(budget, max(3, min(20, budget // 5)))
        stagnation_limit = max(1, budget // 10)
        stagnation_counter = 0

        # Initialize population
        pop = lb + (ub - lb) * rng.rand(pop_size, self.dim)
        pop_fit = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= budget:
                break
            pop_fit[i] = func(pop[i])
            evals += 1
            if pop_fit[i] < self.best_val:
                self.best_val = pop_fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)

        # Main loop
        while evals < budget:
            # Restart condition
            if stagnation_counter >= stagnation_limit:
                # Reinitialize half of population randomly
                num_restart = pop_size // 2
                restart_idx = rng.choice(pop_size, num_restart, replace=False)
                for idx in restart_idx:
                    if evals >= budget:
                        break
                    pop[idx] = lb + (ub - lb) * rng.rand(self.dim)
                    pop_fit[idx] = func(pop[idx])
                    evals += 1
                    if pop_fit[idx] < self.best_val:
                        self.best_val = pop_fit[idx]
                        self.best_x = pop[idx].copy()
                        report_best(self.best_val, self.best_x)
                stagnation_counter = 0

            new_pop = pop.copy()
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Choose mutation strategy probabilistically (exploration rate decays)
                exploration_rate = max(0.1, 1.0 - 0.9 * evals / budget)
                if rng.rand() < exploration_rate:
                    # rand/1 (exploration)
                    idxs = [j for j in range(pop_size) if j != i]
                    a, b, c = rng.choice(idxs, 3, replace=False)
                    F = 0.5 + 0.5 * rng.rand()
                    mutant = pop[a] + F * (pop[b] - pop[c])
                else:
                    # current-to-best/1 (exploitation)
                    idxs = [j for j in range(pop_size) if j != i]
                    a, b = rng.choice(idxs, 2, replace=False)
                    F = 0.8
                    mutant = pop[i] + F * (self.best_x - pop[i]) + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover
                CR = 0.9
                cross_points = rng.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.randint(self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_fit = func(trial)
                evals += 1
                if trial_fit < self.best_val:
                    self.best_val = trial_fit
                    self.best_x = trial.copy()
                    report_best(self.best_val, self.best_x)
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                # Greedy selection
                if trial_fit < pop_fit[i]:
                    new_pop[i] = trial
                    pop_fit[i] = trial_fit
            pop = new_pop

        return self.best_val, self.best_x