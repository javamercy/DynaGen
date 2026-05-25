import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # population size: at least 3, at most budget, scaled with dimension
        self.popsize = min(budget, max(3, min(4 * dim, budget // 2)))
        # stall limit: fraction of budget, at least 1
        self.stall_limit = max(1, budget // 20)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize
        # initialize population
        pop = self.rng.uniform(lb, ub, size=(popsize, dim))
        fitness = np.full(popsize, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0
        # evaluate initial population
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
        # ensure at least one evaluation was done
        if best_x is None:
            x = self.rng.uniform(lb, ub)
            val = func(x)
            evaluations += 1
            best_val = val
            best_x = x.copy()
            report_best(best_val, best_x)
        # main DE loop
        generations_since_improvement = 0
        while evaluations < self.budget:
            # dither F and CR per generation
            F = self.rng.uniform(0.5, 1.0)
            CR = self.rng.uniform(0.5, 1.0)
            for i in range(popsize):
                if evaluations >= self.budget:
                    break
                # select best index
                idx_best = np.argmin(fitness)
                # distinct random indices not equal to i
                candidates = [j for j in range(popsize) if j != i]
                r1, r2 = self.rng.choice(candidates, 2, replace=False)
                # current-to-best/1 mutation
                mutant = pop[i] + F * (pop[idx_best] - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
                cross_points = self.rng.random(dim) < CR
                if not np.any(cross_points):
                    cross_points[self.rng.integers(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evaluations += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        generations_since_improvement = 0
                    else:
                        generations_since_improvement += 1
                else:
                    generations_since_improvement += 1
                # check stall limit within loop? keep outside for simplicity
            # restart if stalled
            if generations_since_improvement > self.stall_limit:
                n_restart = popsize // 2
                restart_indices = self.rng.choice(popsize, n_restart, replace=False)
                for idx in restart_indices:
                    if evaluations >= self.budget:
                        break
                    new_x = self.rng.uniform(lb, ub)
                    val = func(new_x)
                    evaluations += 1
                    pop[idx] = new_x
                    fitness[idx] = val
                    if val < best_val:
                        best_val = val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)
                generations_since_improvement = 0
        return best_val, best_x