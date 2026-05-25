import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(10, min(4 * dim, budget // 10))
        self.stall_limit = max(10, budget // 30)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize
        pop = self.rng.uniform(lb, ub, size=(popsize, dim))
        fitness = np.full(popsize, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0
        for i in range(popsize):
            if evaluations >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evaluations += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        if best_x is None:
            best_x = self.rng.uniform(lb, ub)
            best_val = func(best_x)
            evaluations += 1
            report_best(best_val, best_x)
        stall_count = 0
        while evaluations < self.budget:
            F = 0.5
            CR = 0.5
            for i in range(popsize):
                if evaluations >= self.budget:
                    break
                idx_best = np.argmin(fitness)
                candidates = [j for j in range(popsize) if j != i]
                r1, r2 = self.rng.choice(candidates, 2, replace=False)
                mutant = pop[i] + F * (pop[idx_best] - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                n = self.rng.integers(dim)
                L = 0
                while self.rng.random() < CR and L < dim:
                    L += 1
                trial = pop[i].copy()
                for j in range(L):
                    trial[(n + j) % dim] = mutant[(n + j) % dim]
                val = func(trial)
                evaluations += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        stall_count = 0
                    else:
                        stall_count += 1
                else:
                    stall_count += 1
            if stall_count > self.stall_limit:
                sorted_idx = np.argsort(fitness)
                keep = sorted_idx[0]
                worst_indices = sorted_idx[popsize // 2:]
                std = 0.05 * (ub - lb)
                for idx in worst_indices:
                    if idx == keep:
                        continue
                    if evaluations >= self.budget:
                        break
                    new_x = best_x + self.rng.normal(0, std, size=dim)
                    new_x = np.clip(new_x, lb, ub)
                    val = func(new_x)
                    evaluations += 1
                    pop[idx] = new_x
                    fitness[idx] = val
                    if val < best_val:
                        best_val = val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)
                stall_count = 0
        return best_val, best_x