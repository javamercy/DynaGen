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

        # Population size
        pop_size = min(budget // 2, max(3 * dim, min(15 + int(dim**0.5), budget // 2)))
        pop_size = max(pop_size, 2 * dim)
        pop_size = min(pop_size, budget)
        pop_size = max(pop_size, 5)  # ensure minimum

        # Latin Hypercube initial population
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

        # Reserve for local search
        reserve = max(2 * dim, 20)
        reserve = min(reserve, budget - evals)
        max_gen = (budget - evals - reserve) // pop_size
        max_gen = max(0, max_gen)

        # Adaptive DE
        for gen in range(max_gen):
            if evals >= budget:
                break
            frac = gen / max_gen if max_gen > 0 else 0.0
            F = 0.8 - 0.3 * frac
            CR = 0.6 + 0.3 * frac
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

        # Local search with restarts
        remaining = budget - evals
        if remaining > 0:
            # Start from best point
            x_curr = best_x.copy()
            f_curr = best_val
            step = 0.1 * (ub - lb)
            min_step = 1e-7 * (ub - lb)
            stagnation = 0
            max_stag = max(5, remaining // 20)
            last_restart = 0

            while evals < budget and np.any(step > min_step):
                improved = False
                for d in range(dim):
                    if evals >= budget:
                        break
                    # positive step
                    x_new = x_curr.copy()
                    x_new[d] = np.clip(x_new[d] + step[d], lb[d], ub[d])
                    val = func(x_new)
                    evals += 1
                    if val < f_curr:
                        x_curr = x_new
                        f_curr = val
                        improved = True
                        if f_curr < best_val:
                            best_val = f_curr
                            best_x = x_curr.copy()
                            report_best(best_val, best_x)
                        break
                    # negative step
                    x_new = x_curr.copy()
                    x_new[d] = np.clip(x_new[d] - step[d], lb[d], ub[d])
                    val = func(x_new)
                    evals += 1
                    if val < f_curr:
                        x_curr = x_new
                        f_curr = val
                        improved = True
                        if f_curr < best_val:
                            best_val = f_curr
                            best_x = x_curr.copy()
                            report_best(best_val, best_x)
                        break
                if improved:
                    stagnation = 0
                    # expand step if consecutive successes
                    step = np.clip(step * 1.5, min_step, ub - lb)
                else:
                    stagnation += 1
                    step = np.clip(step * 0.5, min_step, ub - lb)

                # Random perturbation occasionally
                if evals < budget and rng.uniform() < 0.1:
                    pert = rng.randn(dim) * 0.1 * (ub - lb)
                    x_new = np.clip(x_curr + pert, lb, ub)
                    val = func(x_new)
                    evals += 1
                    if val < f_curr:
                        x_curr = x_new
                        f_curr = val
                        if f_curr < best_val:
                            best_val = f_curr
                            best_x = x_curr.copy()
                            report_best(best_val, best_x)
                        stagnation = 0

                # Stagnation restart
                if stagnation >= max_stag and evals < budget:
                    # reinitialize from best with perturbation
                    scale = rng.uniform(0.1, 0.5)
                    x_new = np.clip(best_x + scale * rng.randn(dim) * (ub - lb), lb, ub)
                    val = func(x_new)
                    evals += 1
                    if val < f_curr:
                        x_curr = x_new
                        f_curr = val
                        if f_curr < best_val:
                            best_val = f_curr
                            best_x = x_curr.copy()
                            report_best(best_val, best_x)
                        stagnation = 0
                    else:
                        # also reset step
                        step = 0.1 * (ub - lb)
                        stagnation = 0
                        last_restart = evals

        return best_val, best_x

    def _latin_hypercube(self, n, d, rng):
        intervals = np.linspace(0, 1, n + 1)
        lhs = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            for i in range(n):
                lhs[i, j] = intervals[perm[i]] + rng.uniform(0, 1 / n)
        return lhs