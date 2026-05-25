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

        # Population size: at least 4*dim, minimum 3, but not too large
        pop_size = max(4 * dim, 3)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)

        # Initialize population uniformly
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        # Evaluate initial population
        for i in range(pop_size):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_val:
                best_val = fitness[i]
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # DE parameters (slightly lower F for exploitation)
        F = 0.7
        CR = 0.8

        # Main DE loop with interspersed local search
        while evals < budget:
            # DE generation
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                ids = rng.choice(candidates, size=3, replace=False)
                a, b, c = ids
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    fitness[i] = trial_fit
                    pop[i] = trial
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # Local search around current best (intensify)
            if evals < budget:
                # Determine scale: start larger, decrease as budget runs out
                remaining = budget - evals
                steps = min(3, remaining)
                scale0 = 0.05 * (ub - lb)
                for _ in range(steps):
                    if evals >= budget:
                        break
                    perturbation = scale0 * rng.randn(dim)
                    candidate = np.clip(best_x + perturbation, lb, ub)
                    cand_fit = func(candidate)
                    evals += 1
                    if cand_fit < best_val:
                        best_val = cand_fit
                        best_x = candidate.copy()
                        report_best(best_val, best_x)

        return best_val, best_x