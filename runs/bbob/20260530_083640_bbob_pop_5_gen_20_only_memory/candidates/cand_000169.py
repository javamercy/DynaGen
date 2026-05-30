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

        # Population size: moderate, scaled by dim
        pop_size = max(4 * dim, min(20 + int(dim**0.5), budget // 3))
        pop_size = min(pop_size, budget)
        if pop_size < 2:
            pop_size = 2

        # LHS initial population
        lhs = self._latin_hypercube(pop_size, dim, rng)
        pop = lb + (ub - lb) * lhs

        pop_fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            evals += 1
            pop_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            if evals >= budget:
                return best_val, best_x

        # Reserve budget for local search
        reserve = max(2 * dim, 20)
        reserve = min(reserve, budget - evals)
        de_budget = budget - evals - reserve
        if de_budget < 0:
            de_budget = 0
        max_gen = de_budget // pop_size if de_budget > 0 else 0

        stagnation_gen = 0
        stag_limit = max(3, max_gen // 4) if max_gen > 0 else 1
        last_best_val = best_val

        for gen in range(max_gen):
            frac = gen / max_gen if max_gen > 0 else 0.0
            F = 0.9 - 0.4 * frac
            CR = 0.5 + 0.4 * frac
            for i in range(pop_size):
                if evals >= budget - reserve:
                    break
                # rand/2 mutation
                indices = [j for j in range(pop_size) if j != i]
                rng.shuffle(indices)
                a, b, c, d, e = indices[:5]
                mutant = pop[a] + F * (pop[b] - pop[c]) + F * (pop[d] - pop[e])
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
                if evals >= budget - reserve:
                    break

            if evals >= budget - reserve:
                break

            # Stagnation check
            if best_val < last_best_val:
                stagnation_gen = 0
                last_best_val = best_val
            else:
                stagnation_gen += 1
            if stagnation_gen >= stag_limit and evals < budget - reserve:
                worst_idx = np.argsort(pop_fitness)[-max(1, pop_size // 4):]
                for idx in worst_idx:
                    if evals >= budget - reserve:
                        break
                    new_x = lb + rng.rand(dim) * (ub - lb)
                    new_val = func(new_x)
                    evals += 1
                    pop[idx] = new_x
                    pop_fitness[idx] = new_val
                    if new_val < best_val:
                        best_val = new_val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)
                stagnation_gen = 0
                last_best_val = best_val

        # Local perturbation phase
        remaining = budget - evals
        if remaining > 0:
            step = 0.1 * (ub - lb)
            for _ in range(remaining):
                if evals >= budget:
                    break
                direction = rng.randn(dim)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                trial = np.clip(best_x + step * direction, lb, ub)
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    report_best(best_val, best_x)
                    # Expand step on success
                    step = np.minimum(step * 1.2, ub - lb)
                else:
                    # Shrink step on failure
                    step = np.maximum(step * 0.9, (ub - lb) * 1e-10)
                # Random restart if not improving
                if evals % (dim * 2) == 0 and evals > 10:
                    center = best_x.copy()
                    new_x = center + 0.5 * (ub - lb) * rng.randn(dim)
                    new_x = np.clip(new_x, lb, ub)
                    # Only if we have budget left
                    if evals < budget:
                        val_new = func(new_x)
                        evals += 1
                        if val_new < best_val:
                            best_val = val_new
                            best_x = new_x.copy()
                            report_best(best_val, best_x)
        return best_val, best_x

    def _latin_hypercube(self, n, d, rng):
        intervals = np.linspace(0, 1, n + 1)
        lhs = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            for i in range(n):
                lhs[i, j] = intervals[perm[i]] + rng.uniform(0, 1 / n)
        return lhs