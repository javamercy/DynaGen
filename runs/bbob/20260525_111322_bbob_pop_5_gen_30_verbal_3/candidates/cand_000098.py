import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        pop_size = max(4, min(4 * dim, budget // 2))
        if pop_size < 2:
            pop_size = min(2, budget)
        if pop_size == 0:
            x = lb + rng.rand(dim) * (ub - lb)
            f = func(x)
            report_best(f, x)
            return f, x

        # Initialize population uniformly
        points = rng.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0

        for i in range(pop_size):
            if evals >= budget:
                break
            x = points[i]
            f = func(x)
            evals += 1
            fitness[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        if evals == 0:
            x = lb + rng.rand(dim) * (ub - lb)
            f = func(x)
            report_best(f, x)
            return f, x

        # jDE parameters
        F = rng.uniform(0.1, 1.0, pop_size)
        CR = rng.uniform(0, 1.0, pop_size)

        while evals < budget:
            new_pop = []
            new_fit = []
            for i in range(pop_size):
                if evals >= budget:
                    break
                # jDE adaptation
                if rng.rand() < 0.1:
                    F[i] = rng.uniform(0.1, 1.0)
                if rng.rand() < 0.1:
                    CR[i] = rng.uniform(0, 1.0)
                # Mutation: DE/rand/1
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(idxs, 3, replace=False)
                mutant = points[a] + F[i] * (points[b] - points[c])
                # Binomial crossover
                trial = points[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                if f_trial <= fitness[i]:
                    new_pop.append(trial.copy())
                    new_fit.append(f_trial)
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                else:
                    new_pop.append(points[i].copy())
                    new_fit.append(fitness[i])
            if len(new_pop) == pop_size:
                points = np.array(new_pop)
                fitness = np.array(new_fit)
            else:
                # partial generation, keep old
                pass
            if evals >= budget:
                break

        return best_f, best_x