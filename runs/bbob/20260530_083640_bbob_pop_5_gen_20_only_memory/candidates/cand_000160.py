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

        best_val = np.inf
        best_x = None
        evals = 0

        pop_size = max(4 * dim, 20)
        pop_size = min(pop_size, budget)

        def lhs(n, d):
            intervals = np.linspace(0, 1, n + 1)
            lhs_arr = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                for i in range(n):
                    lhs_arr[i, j] = intervals[perm[i]] + rng.uniform(0, 1 / n)
            return lhs_arr

        def clip(x):
            return np.clip(x, lb, ub)

        def evaluate(x):
            nonlocal evals, best_val, best_x
            x = clip(x)
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            return val

        def pattern_search(x0, f0, max_evals):
            step = 0.05 * (ub - lb)
            min_step = 1e-5 * (ub - lb)
            evals_local = 0
            while evals_local < max_evals and np.any(step > min_step):
                improved = False
                for i in range(dim):
                    if evals_local >= max_evals:
                        break
                    x_new = x0.copy()
                    x_new[i] = np.clip(x0[i] + step[i], lb[i], ub[i])
                    f_new = func(x_new)
                    evals_local += 1
                    if f_new < f0:
                        x0 = x_new
                        f0 = f_new
                        improved = True
                        if f0 < best_val:
                            best_val = f0
                            best_x = x0.copy()
                            report_best(best_val, best_x)
                        continue
                    x_new = x0.copy()
                    x_new[i] = np.clip(x0[i] - step[i], lb[i], ub[i])
                    f_new = func(x_new)
                    evals_local += 1
                    if f_new < f0:
                        x0 = x_new
                        f0 = f_new
                        improved = True
                        if f0 < best_val:
                            best_val = f0
                            best_x = x0.copy()
                            report_best(best_val, best_x)
                if not improved:
                    step *= 0.5
            return x0, f0, evals_local

        while evals < budget:
            # Random restart? For first run use LHS, subsequent use uniform
            if evals == 0:
                pop = lhs(pop_size, dim)
                pop = lb + (ub - lb) * pop
            else:
                pop = lb + rng.rand(pop_size, dim) * (ub - lb)
            F = rng.uniform(0.1, 0.9, pop_size)
            CR = rng.uniform(0, 1, pop_size)
            fitness = np.full(pop_size, np.inf)
            for i in range(pop_size):
                fitness[i] = evaluate(pop[i])
                if evals >= budget:
                    return best_val, best_x
            stagnation = 0
            max_stagnation = max(5, dim)
            # Reserve evaluations for pattern search
            reserve = min(budget - evals, max(2 * dim, 20))
            max_gen = (budget - evals - reserve) // pop_size
            max_gen = max(0, max_gen)
            if max_gen == 0 and reserve > 0:
                # Do pattern search directly
                x0 = best_x.copy()
                f0 = best_val
                _, _, used = pattern_search(x0, f0, reserve)
                evals += used
                break
            for gen in range(max_gen):
                if evals >= budget:
                    break
                # For each individual, generate trial
                for i in range(pop_size):
                    if evals >= budget:
                        break
                    # Select three distinct random indices different from i
                    indices = list(range(pop_size))
                    indices.remove(i)
                    rng.shuffle(indices)
                    a, b, c = indices[:3]
                    mutant = pop[a] + F[i] * (pop[b] - pop[c])
                    j_rand = rng.randint(dim)
                    trial = np.where(rng.rand(dim) < CR[i], mutant, pop[i])
                    trial = clip(trial)
                    f_trial = func(trial)
                    evals += 1
                    if f_trial < fitness[i]:
                        pop[i] = trial
                        fitness[i] = f_trial
                        if f_trial < best_val:
                            best_val = f_trial
                            best_x = trial.copy()
                            report_best(best_val, best_x)
                        # Update F, CR
                        with_prob = rng.rand()
                        if with_prob < 0.1:
                            F[i] = rng.uniform(0.1, 0.9)
                            CR[i] = rng.uniform(0, 1)
                    else:
                        with_prob = rng.rand()
                        if with_prob < 0.1:
                            F[i] = rng.uniform(0.1, 0.9)
                            CR[i] = rng.uniform(0, 1)
                # Check improvement
                if evals >= budget:
                    break
                new_best = np.min(fitness)
                if new_best < best_val:
                    stagnation = 0
                else:
                    stagnation += 1
                if stagnation >= max_stagnation:
                    break
            # Local search after DE run
            if evals < budget and reserve > 0:
                x0 = best_x.copy()
                f0 = best_val
                _, _, used = pattern_search(x0, f0, reserve)
                evals += used
            if stagnation >= max_stagnation:
                # Restart with new random population
                continue
            else:
                # If DE finished without stagnation, still break out of while loop to do final local search?
                # Actually we already did local search, so break.
                break
        return best_val, best_x