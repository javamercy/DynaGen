import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(20, min(4 * dim, self.budget // 2))
        self.stall_limit = max(10, budget // 20)

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
            CR = 0.9
            for i in range(popsize):
                if evaluations >= self.budget:
                    break
                F = np.clip(self.rng.standard_cauchy() * 0.1 + 0.7, 0.0, 1.0)
                candidates = [j for j in range(popsize) if j != i]
                r1, r2, r3 = self.rng.choice(candidates, 3, replace=False)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
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
                for idx in worst_indices:
                    if evaluations >= self.budget:
                        break
                    if idx == keep:
                        continue
                    new_x = self.rng.uniform(lb, ub)
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