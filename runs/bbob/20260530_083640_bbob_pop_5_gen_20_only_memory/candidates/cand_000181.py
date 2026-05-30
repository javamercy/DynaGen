import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def _latin_hypercube(self, n, d):
        intervals = np.linspace(0, 1, n + 1)
        lhs = np.zeros((n, d))
        for j in range(d):
            perm = self.rng.permutation(n)
            for i in range(n):
                lhs[i, j] = intervals[perm[i]] + self.rng.uniform(0, 1 / n)
        return lhs

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        best_x = lb + rng.rand(dim) * (ub - lb)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        pop_size = max(3 * dim, 10)
        if pop_size > budget // 2:
            pop_size = max(2, budget // 4)
        use_de = pop_size >= 2 and budget >= 4 * dim

        if use_de:
            lhs = self._latin_hypercube(pop_size, dim)
            pop = lb + (ub - lb) * lhs
            pop_fit = np.full(pop_size, np.inf)
            for i in range(pop_size):
                if evals >= budget:
                    break
                x = np.clip(pop[i], lb, ub)
                val = func(x)
                evals += 1
                pop_fit[i] = val
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)

            de_budget = int(budget * 0.7)
            if de_budget < 2 * dim:
                de_budget = budget
            de_evals = evals
            max_gen = (de_budget - de_evals) // pop_size
            max_gen = max(1, max_gen)

            for gen in range(max_gen):
                if evals >= de_budget:
                    break
                frac = gen / max_gen if max_gen > 0 else 0.0
                F = 0.9 - 0.4 * frac**2
                CR = 0.5 + 0.4 * frac**0.5
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
                    val = func(trial)
                    evals += 1
                    if val < pop_fit[i]:
                        pop[i] = trial
                        pop_fit[i] = val
                        if val < best_val:
                            best_val = val
                            best_x = trial.copy()
                            report_best(best_val, best_x)

        step = 0.2 * (ub - lb).mean()
        while evals < budget:
            improved = False
            perm = rng.permutation(dim)
            for i in perm:
                if evals >= budget:
                    break
                trial = best_x.copy()
                trial[i] = np.clip(best_x[i] + step, lb[i], ub[i])
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial
                    report_best(best_val, best_x)
                    improved = True
                    step *= 1.5
                    break
                trial = best_x.copy()
                trial[i] = np.clip(best_x[i] - step, lb[i], ub[i])
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial
                    report_best(best_val, best_x)
                    improved = True
                    step *= 1.5
                    break
            if improved:
                continue
            # Multiple random direction polls
            for _ in range(3):
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
                    best_x = trial
                    report_best(best_val, best_x)
                    step *= 1.5
                    improved = True
                    break
            if not improved:
                step *= 0.7

        return best_val, best_x