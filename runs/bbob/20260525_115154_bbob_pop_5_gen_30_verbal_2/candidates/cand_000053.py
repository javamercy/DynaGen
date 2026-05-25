import numpy as np
from numpy.random import RandomState

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Population size
        pop_size = max(4 * dim, 4)
        if pop_size > budget // 10:
            pop_size = max(4, budget // 10)
        max_no_improve_evals = max(pop_size * 10, 100)

        best_val = np.inf
        best_x = None
        evals = 0

        def evaluate(x):
            nonlocal evals, best_val, best_x
            if evals >= budget:
                return None
            f = func(x)
            evals += 1
            if f < best_val:
                best_val = f
                best_x = x.copy()
                report_best(best_val, best_x)
            return f

        local_search_evals = min(3, max(1, int(budget * 0.01)))
        local_scale = 0.01

        while evals < budget:
            # Initialize population
            pop = rng.uniform(lb, ub, size=(pop_size, dim))
            fitness = np.full(pop_size, np.inf)

            for i in range(pop_size):
                if evals >= budget:
                    break
                f = evaluate(pop[i])
                if f is None:
                    break
                fitness[i] = f

            if evals >= budget:
                break

            F = 0.7
            CR = 0.5
            last_improvement_evals = evals

            while evals < budget:
                if evals - last_improvement_evals > max_no_improve_evals:
                    break

                # Offspring generation
                for i in range(pop_size):
                    if evals >= budget:
                        break
                    candidates = list(range(pop_size))
                    candidates.remove(i)
                    a, b, c = rng.choice(candidates, size=3, replace=False)
                    mutant = pop[a] + F * (pop[b] - pop[c])
                    mutant = np.clip(mutant, lb, ub)
                    j_rand = rng.randint(dim)
                    trial = pop[i].copy()
                    for j in range(dim):
                        if rng.rand() < CR or j == j_rand:
                            trial[j] = mutant[j]
                    trial_fit = evaluate(trial)
                    if trial_fit is None:
                        break
                    if trial_fit < fitness[i]:
                        fitness[i] = trial_fit
                        pop[i] = trial
                        last_improvement_evals = evals

                if evals >= budget:
                    break

                # Adapt F and CR
                std = np.std(pop, axis=0)
                range_vec = ub - lb
                norm_std = std / (range_vec + 1e-12)
                diversity = np.mean(norm_std)
                F = 0.5 + 0.5 * (1 - diversity)
                CR = 0.2 + 0.6 * diversity
                F = np.clip(F, 0.5, 1.0)
                CR = np.clip(CR, 0.2, 0.8)

                # Local search around best
                if best_x is not None:
                    for _ in range(local_search_evals):
                        if evals >= budget:
                            break
                        step = rng.normal(0, local_scale, dim) * (ub - lb)
                        candidate = best_x + step
                        candidate = np.clip(candidate, lb, ub)
                        f = evaluate(candidate)
                        if f is None:
                            break
                        if f < best_val:
                            last_improvement_evals = evals

        return best_val, best_x