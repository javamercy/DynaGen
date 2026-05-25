import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        if budget >= 50:
            self.pop_size = max(4 * dim, 20)
        else:
            self.pop_size = max(4, int(budget / 5))
        self.F = 0.8
        self.CR = 1.0  # effectively no crossover
        self.stagnation_limit = max(10, 2 * dim)
        self.best_val = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        # Initialize population
        pop = lb + (ub - lb) * self.rng.rand(self.pop_size, self.dim)
        pop_fit = np.full(self.pop_size, np.inf)
        for i in range(self.pop_size):
            if evals >= self.budget:
                break
            pop_fit[i] = func(pop[i])
            evals += 1
            if pop_fit[i] < self.best_val:
                self.best_val = pop_fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
        no_improve = 0
        while evals < self.budget:
            new_pop = pop.copy()
            new_fit = pop_fit.copy()
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                # Select indices for mutation
                idxs = [j for j in range(self.pop_size) if j != i]
                a, b, c, d = self.rng.choice(idxs, 4, replace=False)
                # current-to-rand/1 mutation with jittered F
                F_jitter = self.F * self.rng.uniform(0.9, 1.1)
                mutant = pop[a] + self.rng.uniform(-0.5, 0.5) * (pop[b] - pop[c]) + F_jitter * (pop[d] - pop[i])
                # Reflection for bounds
                mutant = self._reflect(mutant, lb, ub)
                trial = mutant
                trial_fit = func(trial)
                evals += 1
                if trial_fit < self.best_val:
                    self.best_val = trial_fit
                    self.best_x = trial.copy()
                    report_best(self.best_val, self.best_x)
                if trial_fit < pop_fit[i]:
                    new_pop[i] = trial
                    new_fit[i] = trial_fit
            # Check improvement
            min_fit = np.min(new_fit)
            if min_fit < self.best_val:
                no_improve = 0
            else:
                no_improve += 1
            pop = new_pop
            pop_fit = new_fit
            # Diversification restart on stagnation
            if no_improve >= self.stagnation_limit and evals < self.budget:
                best_idx = np.argmin(pop_fit)
                best_point = pop[best_idx].copy()
                best_fit = pop_fit[best_idx]
                # Perturb all except the best
                for i in range(self.pop_size):
                    if i == best_idx:
                        continue
                    if evals >= self.budget:
                        break
                    # Random step with size up to 20% of domain
                    step = (ub - lb) * self.rng.uniform(0.0, 0.2, self.dim) * self.rng.choice([-1, 1], self.dim)
                    new_point = best_point + step
                    new_point = self._reflect(new_point, lb, ub)
                    pop[i] = new_point
                    pop_fit[i] = func(new_point)
                    evals += 1
                    if pop_fit[i] < self.best_val:
                        self.best_val = pop_fit[i]
                        self.best_x = pop[i].copy()
                        report_best(self.best_val, self.best_x)
                # Restore best
                pop[best_idx] = best_point
                pop_fit[best_idx] = best_fit
                no_improve = 0
        return self.best_val, self.best_x

    def _reflect(self, x, lb, ub):
        # Reflect out-of-bounds points back into domain
        out_lower = x < lb
        out_upper = x > ub
        x = np.where(out_lower, 2 * lb - x, x)
        x = np.where(out_upper, 2 * ub - x, x)
        # In case still out after reflection (e.g., near zero), clamp
        x = np.clip(x, lb, ub)
        return x