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

        # Population size: at least 4*dim, but limited to allow restarts
        pop_size = max(4 * dim, 4)
        if pop_size > budget // 10:
            pop_size = max(4, budget // 10)

        # Upper bound on generations per restart
        max_no_improve_evals = max(pop_size * 10, 100)

        best_val = np.inf
        best_x = None
        evals = 0

        # Helper to evaluate and update best
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

        # Main loop with restarts
        while evals < budget:
            # Initialize population uniformly
            pop = rng.uniform(lb, ub, size=(pop_size, dim))
            fitness = np.full(pop_size, np.inf)

            # Evaluate initial population
            for i in range(pop_size):
                if evals >= budget:
                    break
                f = evaluate(pop[i])
                if f is None:
                    break
                fitness[i] = f

            if evals >= budget:
                break

            # DE parameters
            F = 0.7
            CR = 0.5
            last_improvement_evals = evals

            # DE generation loop
            while evals < budget:
                # Check for restart: no improvement for too long
                if evals - last_improvement_evals > max_no_improve_evals:
                    break

                # Generate offspring using DE/rand/1/bin
                for i in range(pop_size):
                    if evals >= budget:
                        break
                    # Choose three distinct individuals
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
                    trial_fit = evaluate(trial)
                    if trial_fit is None:
                        break
                    if trial_fit < fitness[i]:
                        fitness[i] = trial_fit
                        pop[i] = trial
                        if trial_fit < best_val:  # Already updated in evaluate
                            pass  # evaluate already called report_best
                        last_improvement_evals = evals

                if evals >= budget:
                    break

                # Adapt F and CR based on population diversity
                # Compute mean dimension-wise standard deviation normalized by range
                std = np.std(pop, axis=0)
                range_vec = ub - lb
                norm_std = std / (range_vec + 1e-12)
                diversity = np.mean(norm_std)
                # Map diversity to F and CR: low diversity -> increase exploration
                # F in [0.5, 1.0], CR in [0.2, 0.8]
                F = 0.5 + 0.5 * (1 - diversity)  # when diversity low, F high
                CR = 0.2 + 0.6 * diversity        # when diversity low, CR low
                # Clamp
                F = np.clip(F, 0.5, 1.0)
                CR = np.clip(CR, 0.2, 0.8)

            # After DE loop (either budget exhausted or restart triggered)
            if evals >= budget:
                break

        return best_val, best_x