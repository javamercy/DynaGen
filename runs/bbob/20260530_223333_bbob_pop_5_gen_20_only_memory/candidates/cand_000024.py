import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.pop_size = max(4, min(5 * dim, budget // 10))
        self.mean_F = 0.5
        self.mean_CR = 0.9
        self.F_history = []
        self.CR_history = []

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop = self.rng.uniform(lb, ub, size=(self.pop_size, self.dim))
        fit = np.full(self.pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        for i in range(self.pop_size):
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            evals += 1
            fit[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(val, x)
            if evals >= self.budget:
                return best_val, best_x

        while evals < self.budget:
            # Sample F and CR for each individual from adaptive distributions
            # Use normal with std 0.1, truncated to [0.1, 0.9] for F, [0.1, 1.0] for CR
            # If history not empty, use historical means
            if len(self.F_history) > 0:
                mean_F = np.mean(self.F_history)
                mean_CR = np.mean(self.CR_history)
            else:
                mean_F = self.mean_F
                mean_CR = self.mean_CR

            new_F_list = []
            new_CR_list = []
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                # Generate F and CR
                F = np.clip(self.rng.normal(mean_F, 0.1), 0.1, 0.9)
                CR = np.clip(self.rng.normal(mean_CR, 0.1), 0.1, 1.0)
                # Mutation
                indices = [j for j in range(self.pop_size) if j != i]
                if len(indices) < 3:
                    break
                r1, r2, r3 = self.rng.choice(indices, size=3, replace=False)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)
                # Crossover
                cross_points = self.rng.random(self.dim) < CR
                cross_points[self.rng.integers(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < fit[i]:
                    pop[i] = trial
                    fit[i] = val
                    new_F_list.append(F)
                    new_CR_list.append(CR)
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(val, trial)
            # Update histories
            self.F_history.extend(new_F_list)
            self.CR_history.extend(new_CR_list)
            # Keep history size manageable
            if len(self.F_history) > 50:
                self.F_history = self.F_history[-50:]
                self.CR_history = self.CR_history[-50:]
        return best_val, best_x