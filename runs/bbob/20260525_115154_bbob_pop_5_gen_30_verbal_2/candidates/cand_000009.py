import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub

        # population size: at least 4*dim, but cap to budget/2 and at least 3
        pop_size = max(4 * dim, 10)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)
        if pop_size < 3:
            pop_size = 3

        # initialize population
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        for i in range(pop_size):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_val:
                best_val = fitness[i]
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # adaptive step size (scalar)
        scale = 0.2 * np.mean(ub - lb)
        # stagnation counter
        stagnation = 0
        stagnation_limit = max(pop_size, budget // 20)

        while evals < budget:
            # generate one trial for each individual
            for i in range(pop_size):
                if evals >= budget:
                    break
                # choose base: global best with 50% probability, else current individual
                if rng.uniform() < 0.5:
                    base = best_x
                else:
                    base = pop[i]
                # perturb with Gaussian noise
                noise = rng.normal(0, scale, size=dim)
                trial = base + noise
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    fitness[i] = val
                    pop[i] = trial
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                    scale = max(scale * 0.9, 1e-15)
                    stagnation = 0
                else:
                    scale = min(scale * 1.1, np.mean(ub - lb))
                    stagnation += 1

            # restart if stagnation
            if not (evals < budget):
                break
            if stagnation >= stagnation_limit:
                # replace worst half with random points
                worst_indices = np.argsort(fitness)[-pop_size//2:]
                for idx in worst_indices:
                    if evals >= budget:
                        break
                    new_x = rng.uniform(lb, ub)
                    new_val = func(new_x)
                    evals += 1
                    fitness[idx] = new_val
                    pop[idx] = new_x
                    if new_val < best_val:
                        best_val = new_val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)
                scale = 0.2 * np.mean(ub - lb)
                stagnation = 0

        return best_val, best_x