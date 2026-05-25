import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(20, min(4 * dim, self.budget // 2))
        self.stall_limit = max(10, budget // 20)
        self.local_evals = max(5, min(30, budget // 30))
        self.sigma_0 = 0.1  # relative step size for local search

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
        sigma = self.sigma_0 * (ub - lb)  # initial step size per dimension (scalar for simplicity)
        while evaluations < self.budget:
            CR = 1.0  # binomial crossover for exploitation
            for i in range(popsize):
                if evaluations >= self.budget:
                    break
                F = np.clip(self.rng.standard_cauchy() * 0.05 + 0.3, 0.0, 1.0)
                idx_best = np.argmin(fitness)
                candidates = [j for j in range(popsize) if j != i]
                r1, r2 = self.rng.choice(candidates, 2, replace=False)
                mutant = pop[i] + F * (pop[idx_best] - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
                j_rand = self.rng.integers(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if self.rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
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
            # local search when stagnation detected
            if stall_count >= self.stall_limit // 2 and evaluations < self.budget:
                local_evals_used = 0
                sigma_local = sigma.copy() if isinstance(sigma, np.ndarray) else sigma
                while local_evals_used < self.local_evals and evaluations < self.budget:
                    candidate = best_x + sigma_local * self.rng.normal(0, 1, size=dim)
                    candidate = np.clip(candidate, lb, ub)
                    val = func(candidate)
                    evaluations += 1
                    local_evals_used += 1
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)
                        stall_count = 0
                        sigma_local = sigma  # reset step size
                    else:
                        sigma_local *= 0.9
            # restart if stalled too much
            if stall_count > self.stall_limit:
                sorted_idx = np.argsort(fitness)
                keep = sorted_idx[0]
                worst_indices = sorted_idx[popsize // 2:]
                sigma_restart = 0.05 * (ub - lb)
                for idx in worst_indices:
                    if evaluations >= self.budget:
                        break
                    if idx == keep:
                        continue
                    new_x = best_x + sigma_restart * self.rng.uniform(-1, 1, size=dim)
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