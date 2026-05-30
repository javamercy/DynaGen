import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng

        # divide budget between global (DE) and local (NM)
        global_budget = int(budget * 0.7)
        local_budget = budget - global_budget

        # DE parameters
        pop_size = max(10, dim * 5)
        if pop_size > global_budget:
            pop_size = global_budget // 2
        if pop_size < 3:
            pop_size = 3
        max_gen = max(1, global_budget // pop_size)

        # initial population: Latin Hypercube sampling
        points = np.empty((pop_size, dim))
        for d in range(dim):
            perm = rng.permutation(pop_size)
            points[:, d] = lb[d] + (perm + 0.5) / pop_size * (ub[d] - lb[d])
        # evaluate
        fitness = np.empty(pop_size)
        evals = 0
        best_x = None
        best_f = np.inf
        for i in range(pop_size):
            if evals >= budget:
                break
            x = np.clip(points[i], lb, ub)
            f = func(x)
            evals += 1
            fitness[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                # report_best(best_f, best_x)

        # DE generations
        F = 0.5
        CR = 0.9
        for gen in range(max_gen):
            if evals >= global_budget:
                break
            for i in range(pop_size):
                if evals >= global_budget:
                    break
                # mutation: select three distinct random indices
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                mutant = points[r1] + F * (points[r2] - points[r3])
                # clip to bounds
                mutant = np.clip(mutant, lb, ub)
                # crossover
                j_rand = rng.integers(dim)
                trial = np.where(rng.random(dim) < CR, mutant, points[i])
                trial[j_rand] = mutant[j_rand]
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    points[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        # report_best(best_f, best_x)

        # local refinement with Nelder-Mead
        if local_budget > 0 and best_x is not None:
            # wrap func to count calls
            calls = 0
            def wrapped_func(x):
                nonlocal calls
                calls += 1
                return func(x)
            res = minimize(wrapped_func, best_x, method='Nelder-Mead',
                           options={'maxfev': local_budget, 'xatol':1e-8, 'fatol':1e-8},
                           bounds=[(lb[i], ub[i]) for i in range(dim)])
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
                # report_best(best_f, best_x)

        # final call to report_best if best_x exists
        if best_x is not None:
            pass # report_best already called on improvements
        return best_f, best_x