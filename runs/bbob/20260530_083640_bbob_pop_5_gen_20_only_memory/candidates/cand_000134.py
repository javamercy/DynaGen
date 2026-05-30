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

        # Initial random point
        best_x = lb + rng.rand(dim) * (ub - lb)
        best_f = func(best_x)
        evals = 1
        report_best(best_f, best_x)

        # Population size for DE
        pop_size = min(budget // 2, max(3 * dim, min(15 + int(dim**0.5), budget // 2)))
        pop_size = max(pop_size, 2 * dim)
        pop_size = min(pop_size, budget - evals)

        if pop_size <= 0:
            # Not enough budget for DE, just do local search
            return self._local_search(func, best_x, best_f, evals, budget, lb, ub, rng)

        # Latin Hypercube initial population
        lhs = self._latin_hypercube(pop_size, dim, rng)
        bounds_arr = np.array([lb, ub]).T
        pop = bounds_arr[:, 0] + (bounds_arr[:, 1] - bounds_arr[:, 0]) * lhs

        pop_fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            evals += 1
            pop_fitness[i] = val
            if val < best_f:
                best_f = val
                best_x = x.copy()
                report_best(best_f, best_x)
            if evals >= budget:
                return best_f, best_x

        # Reserve for local search (at least 2*dim or 20)
        reserve_local = max(2 * dim, 20)
        reserve_local = min(reserve_local, budget - evals)
        max_gen = (budget - evals - reserve_local) // pop_size
        max_gen = max(0, max_gen)

        # Stagnation detection
        stagnation_gen = 0
        stag_limit = max(5, max_gen // 5) if max_gen > 0 else 1
        last_best_val = best_f

        # DE loop
        for gen in range(max_gen):
            frac = gen / max_gen if max_gen > 0 else 0.0
            F = 0.5 + 0.4 * frac
            CR = 0.9 - 0.4 * frac
            for i in range(pop_size):
                indices = [j for j in range(pop_size) if j != i]
                rng.shuffle(indices)
                a, b, c = indices[:3]
                jitter = 0.001 * rng.randn(dim)
                mutant = pop[a] + (F + jitter) * (pop[b] - pop[c])
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                    if val < best_f:
                        best_f = val
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                if evals >= budget:
                    return best_f, best_x

            # Stagnation check
            if best_f < last_best_val:
                stagnation_gen = 0
                last_best_val = best_f
            else:
                stagnation_gen += 1

            if stagnation_gen >= stag_limit and evals < budget:
                # Reinitialize worst 25% with Gaussian perturbations around best
                worst_idx = np.argsort(pop_fitness)[-max(1, pop_size // 4):]
                scale = 0.2 * (ub - lb)
                for idx in worst_idx:
                    if evals >= budget:
                        break
                    new_x = best_x + scale * rng.randn(dim)
                    new_x = np.clip(new_x, lb, ub)
                    new_val = func(new_x)
                    evals += 1
                    pop[idx] = new_x
                    pop_fitness[idx] = new_val
                    if new_val < best_f:
                        best_f = new_val
                        best_x = new_x.copy()
                        report_best(best_f, best_x)
                stagnation_gen = 0
                last_best_val = best_f

        # Local search phase
        remaining = budget - evals
        if remaining > 0:
            best_f, best_x = self._local_search(func, best_x, best_f, evals, budget, lb, ub, rng)

        return best_f, best_x

    def _local_search(self, func, best_x, best_f, evals, budget, lb, ub, rng):
        dim = len(best_x)
        step = 0.2 * (ub - lb)
        min_step = 1e-5 * (ub - lb)
        fail_counter = 0
        restart_interval = max(1, (budget - evals) // 5) if budget - evals > 0 else 1
        last_restart_evals = evals

        while evals < budget:
            # Scheduled restart
            if evals - last_restart_evals >= restart_interval and evals < budget:
                new_x = lb + rng.rand(dim) * (ub - lb)
                new_f = func(new_x)
                evals += 1
                if new_f < best_f:
                    best_f = new_f
                    best_x = new_x.copy()
                    report_best(best_f, best_x)
                step = 0.2 * (ub - lb)
                fail_counter = 0
                last_restart_evals = evals
                continue

            success = False
            perm = rng.permutation(dim)
            for i in perm:
                if evals >= budget:
                    break
                # Positive direction
                trial = best_x.copy()
                trial[i] = np.clip(best_x[i] + step[i], lb[i], ub[i])
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step[i] = min(step[i] * 2, ub[i] - lb[i])
                    success = True
                    fail_counter = 0
                    break
                # Negative direction
                trial[i] = np.clip(best_x[i] - step[i], lb[i], ub[i])
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step[i] = min(step[i] * 2, ub[i] - lb[i])
                    success = True
                    fail_counter = 0
                    break
                else:
                    step[i] = max(step[i] * 0.5, min_step[i])

            if not success:
                # Random perturbation every 5 fails
                if fail_counter % 5 == 0 and evals < budget:
                    scale = 0.1 * (ub - lb)
                    perturbation = scale * rng.randn(dim)
                    trial = np.clip(best_x + perturbation, lb, ub)
                    f = func(trial)
                    evals += 1
                    if f < best_f:
                        best_f = f
                        best_x = trial
                        report_best(best_f, best_x)
                        step = np.minimum(step * 2, ub - lb)
                        success = True
                        fail_counter = 0
                if not success:
                    fail_counter += 1

            # If step sizes become too small, restart from a random point
            if np.all(step <= min_step) and evals < budget:
                new_x = lb + rng.rand(dim) * (ub - lb)
                new_f = func(new_x)
                evals += 1
                if new_f < best_f:
                    best_f = new_f
                    best_x = new_x.copy()
                    report_best(best_f, best_x)
                step = 0.2 * (ub - lb)
                fail_counter = 0
                last_restart_evals = evals

        return best_f, best_x

    def _latin_hypercube(self, n, d, rng):
        intervals = np.linspace(0, 1, n + 1)
        lhs = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            for i in range(n):
                lhs[i, j] = intervals[perm[i]] + rng.uniform(0, 1 / n)
        return lhs