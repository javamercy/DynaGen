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

        # Local pattern search from a starting point
        def local_pattern_search(start, step0, max_evals):
            nonlocal evals, best_val, best_x
            x = start.copy()
            step = step0
            while evals < budget and max_evals > 0:
                improved = False
                for j in range(dim):
                    if evals >= budget or max_evals <= 0:
                        break
                    # Try positive direction
                    x_try = x.copy()
                    x_try[j] = min(x_try[j] + step, ub[j])
                    if x_try[j] <= ub[j]:
                        f = evaluate(x_try)
                        max_evals -= 1
                        if f is not None and f < best_val:
                            x[j] = x_try[j]
                            improved = True
                            break
                    # Try negative direction
                    x_try = x.copy()
                    x_try[j] = max(x_try[j] - step, lb[j])
                    if x_try[j] >= lb[j]:
                        f = evaluate(x_try)
                        max_evals -= 1
                        if f is not None and f < best_val:
                            x[j] = x_try[j]
                            improved = True
                            break
                if not improved:
                    step /= 2.0
                    if step < 1e-10:
                        break
            return x

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
                        last_improvement_evals = evals  # updated in evaluate but set here too

                if evals >= budget:
                    break

                # Adapt F and CR based on population diversity
                std = np.std(pop, axis=0)
                range_vec = ub - lb
                norm_std = std / (range_vec + 1e-12)
                diversity = np.mean(norm_std)
                F = 0.5 + 0.5 * (1 - diversity)
                CR = 0.2 + 0.6 * diversity
                F = np.clip(F, 0.5, 1.0)
                CR = np.clip(CR, 0.2, 0.8)

            # After DE loop, perform local search from best point
            if evals < budget and best_x is not None:
                # Allocate up to 50 evaluations for local search, but also scale with dim
                local_budget = min(50, (budget - evals) // 2)
                if local_budget > 0:
                    step0 = 0.1 * (ub - lb).mean()
                    best_x = local_pattern_search(best_x, step0, local_budget)
                # Prepare for next restart: reinitialize population around best point
                if evals < budget:
                    spread = 0.1 * (ub - lb)  # 10% of range
                    for i in range(pop_size):
                        new_x = best_x + rng.normal(0, spread, size=dim)
                        new_x = np.clip(new_x, lb, ub)
                        pop[i] = new_x
                        f = evaluate(new_x)
                        if evals >= budget:
                            break
                        fitness[i] = f if f is not None else np.inf
                # Reset last_improvement_evals for next DE loop
                last_improvement_evals = evals
                # Note: we reuse the same pop variable; the next restart will reinitialize anyway

        return best_val, best_x