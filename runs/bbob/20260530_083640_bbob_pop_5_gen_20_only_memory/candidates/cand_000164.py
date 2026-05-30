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

        best_val = None
        best_x = None
        evals = 0

        # Ensure at least one evaluation
        if budget == 0:
            return None, None

        # Initial random point
        x0 = lb + rng.rand(dim) * (ub - lb)
        val0 = func(x0)
        evals += 1
        best_val = val0
        best_x = x0.copy()
        report_best(best_val, best_x)

        if budget <= 2 * dim + 5:
            # Small budget: random search + coordinate search
            n_random = min(budget - evals, max(1, dim))
            for _ in range(n_random):
                if evals >= budget:
                    break
                x = lb + rng.rand(dim) * (ub - lb)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            self._coordinate_search(func, lb, ub, dim, budget, rng, best_val, best_x, evals)
            return best_val, best_x

        # Population size
        pop_size = min(max(3 * dim, 20), budget // 2)
        pop_size = max(pop_size, 2)  # need at least 2 for DE

        # Reserve for local search
        reserve = min(2 * dim, 20)
        max_gen = (budget - evals - reserve) // pop_size
        if max_gen < 0:
            max_gen = 0

        # Initialize population uniformly
        pop = lb + (ub - lb) * rng.rand(pop_size, dim)
        pop_fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            pop_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        if evals >= budget:
            # Run local search with remaining budget (might be 0)
            self._coordinate_search(func, lb, ub, dim, budget, rng, best_val, best_x, evals)
            return best_val, best_x

        # DE iterations
        for gen in range(max_gen):
            if evals >= budget:
                break
            frac = gen / max_gen if max_gen > 0 else 0.0
            F = 0.9 - 0.4 * frac
            CR = 0.5 + 0.4 * frac
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Select three distinct indices not equal to i
                candidates = [j for j in range(pop_size) if j != i]
                rng.shuffle(candidates)
                a, b, c = candidates[:3]
                mutant = pop[a] + F * (pop[b] - pop[c])
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        # Local search from best point
        self._coordinate_search(func, lb, ub, dim, budget, rng, best_val, best_x, evals)
        return best_val, best_x

    def _coordinate_search(self, func, lb, ub, dim, budget, rng, best_val, best_x, evals):
        if evals >= budget:
            return best_val, best_x
        step = 0.1 * (ub - lb)
        # Continue until budget exhausted
        while evals < budget:
            success = False
            perm = rng.permutation(dim)
            for i in perm:
                if evals >= budget:
                    break
                # Positive step
                trial = best_x.copy()
                trial[i] = np.clip(best_x[i] + step[i], lb[i], ub[i])
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial
                    report_best(best_val, best_x)
                    step[i] = min(step[i] * 2, ub[i] - lb[i])
                    success = True
                    break
                # Negative step
                trial[i] = np.clip(best_x[i] - step[i], lb[i], ub[i])
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial
                    report_best(best_val, best_x)
                    step[i] = min(step[i] * 2, ub[i] - lb[i])
                    success = True
                    break
                else:
                    step[i] = max(step[i] * 0.5, (ub[i] - lb[i]) * 1e-15)
            if not success and evals < budget:
                # Random direction move
                direction = rng.randn(dim)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                trial = np.clip(best_x + step * direction, lb, ub)
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial
                    report_best(best_val, best_x)
                    step = np.minimum(step * 2, ub - lb)
        return best_val, best_x