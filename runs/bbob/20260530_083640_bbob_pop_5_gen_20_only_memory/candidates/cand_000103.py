import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = np.random.RandomState(self.seed)

        best_val = np.inf
        best_x = None
        evals = 0

        def evaluate(x):
            nonlocal evals, best_val, best_x
            if evals >= budget:
                return None
            x_clipped = np.clip(x, lb, ub)
            val = func(x_clipped)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x_clipped.copy()
                report_best(best_val, best_x)
            return val

        # Reserve evaluations for local search (larger fraction for exploitation)
        reserve_local = max(4 * dim, 20)
        if reserve_local > budget:
            reserve_local = budget
        de_budget = budget - reserve_local
        if de_budget < 2:
            de_budget = max(1, budget // 2)
            reserve_local = budget - de_budget

        # LHS initialization
        pop_size = max(2 * dim, min(20, de_budget // 2))
        if pop_size * 3 > de_budget:
            pop_size = max(2, de_budget // 3)
        max_gens = de_budget // pop_size if pop_size > 0 else 0

        def lhs(n, d):
            intervals = np.linspace(0, 1, n + 1)
            samples = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                for i in range(n):
                    samples[i, j] = intervals[perm[i]] + rng.uniform(0, 1/n)
            return samples

        pop = lb + (ub - lb) * lhs(pop_size, dim)
        pop_fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= de_budget:
                break
            val = evaluate(pop[i])
            if val is None:
                break
            pop_fitness[i] = val

        # DE with adaptive F and CR
        for gen in range(max_gens):
            if evals >= de_budget:
                break
            progress = gen / max_gens if max_gens > 0 else 0
            F = 0.8 - 0.6 * progress
            CR = 0.9 - 0.4 * progress
            for i in range(pop_size):
                if evals >= de_budget:
                    break
                indices = [j for j in range(pop_size) if j != i]
                rng.shuffle(indices)
                a, b, c = indices[:3]
                mutant = pop[a] + F * (pop[b] - pop[c])
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = evaluate(trial)
                if val is None:
                    break
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val

        # Intensified local search
        if evals < budget and best_x is not None:
            # Initialize step sizes and directions
            step = 0.1 * (ub - lb)
            step_min = 1e-5 * (ub - lb)
            # Precompute directions: axes and random
            axes = np.eye(dim)
            num_random = dim  # number of random directions
            random_dirs = rng.randn(num_random, dim)
            random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
            directions = np.vstack([axes, random_dirs])
            n_dirs = directions.shape[0]
            # Local search loop
            for iteration in range(100):
                if evals >= budget:
                    break
                improved_this_iter = False
                for d_idx in range(n_dirs):
                    if evals >= budget:
                        break
                    dir_vec = directions[d_idx]
                    # Positive step
                    x_new = best_x + step * dir_vec
                    x_new = np.clip(x_new, lb, ub)
                    val = evaluate(x_new)
                    if val is None:
                        break
                    if val < best_val:
                        best_x = x_new.copy()
                        best_val = val
                        improved_this_iter = True
                        step = step * 1.2
                        continue
                    # Negative step
                    x_new = best_x - step * dir_vec
                    x_new = np.clip(x_new, lb, ub)
                    val = evaluate(x_new)
                    if val is None:
                        break
                    if val < best_val:
                        best_x = x_new.copy()
                        best_val = val
                        improved_this_iter = True
                        step = step * 1.2
                    else:
                        step = step * 0.5
                if not improved_this_iter:
                    # Reduce step globally
                    step = step * 0.9
                if np.all(step < step_min):
                    break

        return best_val, best_x