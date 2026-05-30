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

        # ---------- Initial LHS sampling ----------
        pop_size = min(budget // 2, max(3 * dim, min(15 + int(dim**0.5), budget // 2)))
        pop_size = max(pop_size, 2 * dim)
        pop_size = min(pop_size, budget)

        lhs = self._latin_hypercube(pop_size, dim, rng)
        pop = lb + (ub - lb) * lhs

        best_val = np.inf
        best_x = None
        evals = 0
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

        # ---------- Adaptive DE ----------
        reserve = max(2 * dim, 20)
        reserve = min(reserve, budget - evals)
        max_gen = (budget - evals - reserve) // pop_size
        max_gen = max(0, max_gen)

        for gen in range(max_gen):
            frac = gen / max_gen if max_gen > 0 else 0.0
            F = 0.9 - 0.4 * frac
            CR = 0.5 + 0.4 * frac
            for i in range(pop_size):
                if evals >= budget:
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
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        # ---------- Coordinate search refinement ----------
        if evals < budget:
            x = best_x.copy()
            step = 0.1 * (ub - lb)
            stagnation = 0
            max_stag = max(1, budget // 10)
            # No scheduled restart, only stagnation restart
            while evals < budget:
                success = False
                perm = rng.permutation(dim)
                for d in perm:
                    if evals >= budget:
                        break
                    # Positive direction
                    trial = x.copy()
                    trial[d] = np.clip(x[d] + step[d], lb[d], ub[d])
                    val = func(trial)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        x = trial.copy()
                        step[d] = min(step[d] * 2, ub[d] - lb[d])
                        success = True
                        stagnation = 0
                        break
                    # Negative direction
                    trial = x.copy()
                    trial[d] = np.clip(x[d] - step[d], lb[d], ub[d])
                    val = func(trial)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        x = trial.copy()
                        step[d] = min(step[d] * 2, ub[d] - lb[d])
                        success = True
                        stagnation = 0
                        break
                    else:
                        step[d] = max(step[d] * 0.5, (ub[d] - lb[d]) * 1e-10)
                if not success and evals < budget:
                    # Random direction search
                    direction = rng.randn(dim)
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                    trial = np.clip(x + step * direction, lb, ub)
                    val = func(trial)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        x = trial.copy()
                        step = np.minimum(step * 2, ub - lb)
                        success = True
                        stagnation = 0
                    else:
                        stagnation += 1
                # Random perturbation with 20% probability
                if evals < budget and rng.uniform() < 0.2:
                    scale = rng.uniform(0.1, 0.5)
                    perturbation = scale * (ub - lb) * rng.randn(dim)
                    trial = np.clip(x + perturbation, lb, ub)
                    val = func(trial)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        x = trial.copy()
                        step = np.minimum(step * 2, ub - lb)
                        success = True
                        stagnation = 0
                # Stagnation restart
                if stagnation >= max_stag and evals < budget:
                    new_x = lb + rng.rand(dim) * (ub - lb)
                    val = func(new_x)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)
                        x = new_x.copy()
                    step = 0.1 * (ub - lb)
                    stagnation = 0

        return best_val, best_x

    def _latin_hypercube(self, n, d, rng):
        intervals = np.linspace(0, 1, n + 1)
        lhs = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            for i in range(n):
                lhs[i, j] = intervals[perm[i]] + rng.uniform(0, 1 / n)
        return lhs