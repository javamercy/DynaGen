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

        # Population size for DE: larger for exploration
        pop_size = min(budget // 2, max(4 * dim, 20))
        pop_size = max(pop_size, 2 * dim)
        pop_size = min(pop_size, budget)

        # Latin Hypercube initial population
        lhs = self._latin_hypercube(pop_size, dim, rng)
        bounds = np.array([lb, ub]).T
        pop = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs

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

        # Reserve for local search: allow more DE generations
        reserve = max(dim, 10)
        reserve = min(reserve, budget - evals)
        max_gen = (budget - evals - reserve) // pop_size
        max_gen = max(0, max_gen)

        # Stagnation detection
        stagnation_gen = 0
        stag_limit = max(5, max_gen // 3) if max_gen > 0 else 1
        last_best_val = best_val

        # DE with constant high F and CR for exploration
        F = 0.9
        CR = 0.9
        for gen in range(max_gen):
            for i in range(pop_size):
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
                if evals >= budget:
                    return best_val, best_x

            # Stagnation: restart worst 30% of population
            if best_val < last_best_val:
                stagnation_gen = 0
                last_best_val = best_val
            else:
                stagnation_gen += 1

            if stagnation_gen >= stag_limit and evals < budget:
                worst_idx = np.argsort(pop_fitness)[-max(1, pop_size // 3):]
                for idx in worst_idx:
                    if evals >= budget:
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

        # Local search: coordinate pattern search with enhanced exploration
        remaining = budget - evals
        if remaining > 0:
            best_local = best_x.copy()
            best_local_val = best_val
            step = 0.3 * (ub - lb)  # larger initial step
            stagnation_local = 0
            max_stag_local = max(1, remaining // 8)
            restart_interval = max(1, remaining // 3)  # more frequent restarts
            last_restart_evals = evals

            while evals < budget:
                # Scheduled restart
                if evals - last_restart_evals >= restart_interval and evals < budget:
                    new_x = lb + rng.rand(dim) * (ub - lb)
                    new_val = func(new_x)
                    evals += 1
                    if new_val < best_local_val:
                        best_local_val = new_val
                        best_local = new_x.copy()
                        report_best(best_local_val, best_local)
                    step = 0.3 * (ub - lb)
                    stagnation_local = 0
                    last_restart_evals = evals
                    continue

                success = False
                perm = rng.permutation(dim)
                for i in perm:
                    if evals >= budget:
                        break
                    # Positive direction
                    trial = best_local.copy()
                    trial[i] = np.clip(best_local[i] + step[i], lb[i], ub[i])
                    val = func(trial)
                    evals += 1
                    if val < best_local_val:
                        best_local_val = val
                        best_local = trial
                        report_best(best_local_val, best_local)
                        step[i] = min(step[i] * 2, ub[i] - lb[i])
                        success = True
                        stagnation_local = 0
                        break
                    # Negative direction
                    trial[i] = np.clip(best_local[i] - step[i], lb[i], ub[i])
                    val = func(trial)
                    evals += 1
                    if val < best_local_val:
                        best_local_val = val
                        best_local = trial
                        report_best(best_local_val, best_local)
                        step[i] = min(step[i] * 2, ub[i] - lb[i])
                        success = True
                        stagnation_local = 0
                        break
                    else:
                        step[i] = max(step[i] * 0.5, (ub[i] - lb[i]) * 1e-10)

                if not success and evals < budget:
                    direction = rng.randn(dim)
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                    trial = np.clip(best_local + step * direction, lb, ub)
                    val = func(trial)
                    evals += 1
                    if val < best_local_val:
                        best_local_val = val
                        best_local = trial
                        report_best(best_local_val, best_local)
                        step = np.minimum(step * 2, ub - lb)
                        success = True
                        stagnation_local = 0
                    else:
                        stagnation_local += 1

                # Random perturbation with 50% probability
                if evals < budget and rng.uniform() < 0.5:
                    scale = rng.uniform(0.1, 0.5)
                    perturbation = scale * (ub - lb) * rng.randn(dim)
                    trial = np.clip(best_local + perturbation, lb, ub)
                    val = func(trial)
                    evals += 1
                    if val < best_local_val:
                        best_local_val = val
                        best_local = trial
                        report_best(best_local_val, best_local)
                        step = np.minimum(step * 2, ub - lb)
                        success = True
                        stagnation_local = 0

                # Stagnation restart
                if stagnation_local >= max_stag_local and evals < budget:
                    new_x = lb + rng.rand(dim) * (ub - lb)
                    new_val = func(new_x)
                    evals += 1
                    if new_val < best_local_val:
                        best_local_val = new_val
                        best_local = new_x.copy()
                        report_best(best_local_val, best_local)
                    step = 0.3 * (ub - lb)
                    stagnation_local = 0
                    last_restart_evals = evals

            if best_local_val < best_val:
                best_val = best_local_val
                best_x = best_local.copy()

        return best_val, best_x

    def _latin_hypercube(self, n, d, rng):
        intervals = np.linspace(0, 1, n + 1)
        lhs = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            for i in range(n):
                lhs[i, j] = intervals[perm[i]] + rng.uniform(0, 1 / n)
        return lhs