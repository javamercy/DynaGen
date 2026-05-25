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
        if self.budget <= 0:
            return np.inf, np.zeros(self.dim)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize
        # Latin hypercube sampling for initialization
        samples = np.zeros((popsize, dim))
        for d in range(dim):
            intervals = np.linspace(0, 1, popsize + 1)
            points = self.rng.uniform(intervals[:-1], intervals[1:])
            self.rng.shuffle(points)
            samples[:, d] = points
        pop = lb + samples * (ub - lb)
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
                idx_best = np.argmin(fitness)
                candidates = [j for j in range(popsize) if j != i]
                r1, r2 = self.rng.choice(candidates, 2, replace=False)
                mutant = pop[i] + F * (pop[idx_best] - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # exponential crossover
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
            # restart if stalled
            if stall_count > self.stall_limit:
                sorted_idx = np.argsort(fitness)
                keep = sorted_idx[0]
                worst_indices = sorted_idx[popsize // 2:]
                # generate new points via LHS for worst half
                new_samples = np.zeros((len(worst_indices), dim))
                for d in range(dim):
                    intervals = np.linspace(0, 1, len(worst_indices) + 1)
                    points = self.rng.uniform(intervals[:-1], intervals[1:])
                    self.rng.shuffle(points)
                    new_samples[:, d] = points
                new_points = lb + new_samples * (ub - lb)
                for idx, new_x in zip(worst_indices, new_points):
                    if evaluations >= self.budget:
                        break
                    if idx == keep:
                        continue
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