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
        budget = self.budget
        rng = self.rng
        evals = 0

        # Initial evaluation of a random point to ensure at least one evaluation
        x0 = lb + (ub - lb) * rng.rand(self.dim)
        f0 = func(x0)
        evals += 1
        self.best_val = f0
        self.best_x = x0.copy()
        report_best(self.best_val, self.best_x)

        if budget <= 1:
            return self.best_val, self.best_x

        # Differential Evolution phase
        pop_size = min(budget // 5, 20)
        if pop_size < 3:
            pop_size = 3
        F = 0.8
        CR = 0.9

        # Initialize population
        pop = lb + (ub - lb) * rng.rand(pop_size, self.dim)
        pop_fit = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            pop_fit[i] = val
            if val < self.best_val:
                self.best_val = val
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)

        # Main DE loop with decreasing population and F, CR
        generation = 0
        while evals < budget and pop_size >= 3:
            new_pop = pop.copy()
            new_fit = pop_fit.copy()
            # Decay parameters over generations
            F_current = 0.8 - 0.4 * (generation / (budget // pop_size + 1))
            CR_current = 0.9 - 0.3 * (generation / (budget // pop_size + 1))
            F_current = max(F_current, 0.4)
            CR_current = max(CR_current, 0.6)

            for i in range(pop_size):
                if evals >= budget:
                    break
                # Mutation: current-to-best/1
                idxs = [j for j in range(pop_size) if j != i]
                a, b = rng.choice(idxs, 2, replace=False)
                mutant = pop[i] + F_current * (self.best_x - pop[i]) + F_current * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover
                cross_points = rng.rand(self.dim) < CR_current
                if not np.any(cross_points):
                    cross_points[rng.randint(self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_fit = func(trial)
                evals += 1
                if trial_fit < self.best_val:
                    self.best_val = trial_fit
                    self.best_x = trial.copy()
                    report_best(self.best_val, self.best_x)
                if trial_fit < pop_fit[i]:
                    new_pop[i] = trial
                    new_fit[i] = trial_fit
            pop = new_pop
            pop_fit = new_fit
            generation += 1
            # Reduce population size every few generations
            if generation % max(1, (budget // pop_size) // 2) == 0 and pop_size > 3:
                # Sort by fitness and keep best half
                order = np.argsort(pop_fit)
                half = pop_size // 2
                pop = pop[order[:half]]
                pop_fit = pop_fit[order[:half]]
                pop_size = half

        # Local refinement phase: pattern search on best_x
        if evals < budget:
            step_size = 0.05 * (ub - lb)  # initial step relative to bounds
            step_size = np.maximum(step_size, 1e-3 * (ub - lb))
            while evals < budget and np.max(step_size) > 1e-8:
                improved = False
                for d in range(self.dim):
                    if evals >= budget:
                        break
                    # Try positive direction
                    x_plus = self.best_x.copy()
                    x_plus[d] += step_size[d]
                    x_plus[d] = np.clip(x_plus[d], lb[d], ub[d])
                    val = func(x_plus)
                    evals += 1
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = x_plus.copy()
                        report_best(self.best_val, self.best_x)
                        improved = True
                        continue
                    # Try negative direction
                    x_minus = self.best_x.copy()
                    x_minus[d] -= step_size[d]
                    x_minus[d] = np.clip(x_minus[d], lb[d], ub[d])
                    val = func(x_minus)
                    evals += 1
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = x_minus.copy()
                        report_best(self.best_val, self.best_x)
                        improved = True
                if not improved:
                    step_size *= 0.5
        return self.best_val, self.best_x