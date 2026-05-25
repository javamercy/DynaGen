import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.popsize = max(4, min(5 * dim, budget // 4))
        self.F = 0.6
        self.CR = 0.5
        self.local_step = 0.1  # initial step size for local walk (fraction of bound range)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        span = ub - lb
        pop = self.rng.uniform(lb, ub, size=(self.popsize, self.dim))
        fitness = np.full(self.popsize, np.inf)
        evals = 0
        best_val = np.inf
        best_x = None

        # initial evaluations
        for i in range(self.popsize):
            if evals >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        generation = 0
        while evals < self.budget:
            # DE generation
            for i in range(self.popsize):
                if evals >= self.budget:
                    break
                idxs = list(range(self.popsize))
                idxs.remove(i)
                a, b, c = self.rng.choice(idxs, 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                cross = self.rng.rand(self.dim) < self.CR
                if not np.any(cross):
                    cross[self.rng.randint(self.dim)] = True
                trial = np.where(cross, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # Local refinement on best solution (random walk with decaying step)
            if evals < self.budget:
                num_local = max(1, self.popsize // 2)
                for _ in range(num_local):
                    if evals >= self.budget:
                        break
                    step = self.local_step * span / (1 + generation)
                    perturbation = self.rng.normal(0, 1, self.dim) * step
                    candidate = np.clip(best_x + perturbation, lb, ub)
                    val = func(candidate)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)
            generation += 1

        return best_val, best_x