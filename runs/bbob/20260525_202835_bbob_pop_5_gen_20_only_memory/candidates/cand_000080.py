import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(5, min(20, budget // 10))
        self.stall_evals = max(self.popsize, budget // 5)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize
        rng = self.rng
        pop = rng.uniform(lb, ub, size=(popsize, dim))
        fitness = np.full(popsize, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0
        for i in range(popsize):
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
        if best_x is None:
            best_x = rng.uniform(lb, ub)
            best_val = func(best_x)
            evals += 1
            report_best(best_val, best_x)
        F = 0.8
        CR = 0.9
        evals_since_improvement = 0
        while evals < self.budget:
            for i in range(popsize):
                if evals >= self.budget:
                    break
                candidates = list(range(popsize))
                candidates.remove(i)
                idx = rng.choice(candidates, 3, replace=False)
                a, b, c = idx
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                cross_mask = rng.random(dim) < CR
                if not np.any(cross_mask):
                    cross_mask[rng.integers(dim)] = True
                trial = np.where(cross_mask, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        evals_since_improvement = 0
                    else:
                        evals_since_improvement += 1
                else:
                    evals_since_improvement += 1
                if evals_since_improvement >= self.stall_evals:
                    n_restart = popsize // 2
                    restart_idx = rng.choice(popsize, n_restart, replace=False)
                    for idx in restart_idx:
                        if evals >= self.budget:
                            break
                        new_x = rng.uniform(lb, ub)
                        val = func(new_x)
                        evals += 1
                        pop[idx] = new_x
                        fitness[idx] = val
                        if val < best_val:
                            best_val = val
                            best_x = new_x.copy()
                            report_best(best_val, best_x)
                    evals_since_improvement = 0
        return best_val, best_x