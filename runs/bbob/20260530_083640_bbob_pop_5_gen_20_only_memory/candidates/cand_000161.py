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

        # Reserve budget for local search: at least 2*dim, up to 1/4 of budget
        ls_budget = max(2 * dim, budget // 4)
        # Population size for DE
        pop_size = max(3 * dim, min(15 + int(dim**0.5), budget // 2))
        pop_size = min(pop_size, budget - ls_budget)
        if pop_size < 1:
            pop_size = 1

        # Latin Hypercube initial population
        intervals = np.linspace(0, 1, pop_size + 1)
        lhs = np.zeros((pop_size, dim))
        for j in range(dim):
            perm = rng.permutation(pop_size)
            for i in range(pop_size):
                lhs[i, j] = intervals[perm[i]] + rng.uniform(0, 1 / pop_size)
        bounds = np.array([lb, ub]).T
        pop = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs

        pop_fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

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

        # DE parameters (fixed)
        F = 0.8
        CR = 0.9
        max_gen = max(0, (budget - evals - ls_budget) // pop_size) if pop_size > 0 else 0
        max_gen = min(max_gen, 200)

        # Stagnation detection
        stagnation_gen = 0
        stag_limit = max(5, max_gen // 5) if max_gen > 0 else 1
        last_best_val = best_val

        for gen in range(max_gen):
            for i in range(pop_size):
                if evals >= budget - ls_budget:
                    break
                # Mutation: best/1
                a, b = rng.choice([j for j in range(pop_size) if j != i], 2, replace=False)
                mutant = best_x + F * (pop[a] - pop[b])
                # Crossover: bin
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
            if evals >= budget - ls_budget:
                break

            # Stagnation check
            if best_val < last_best_val:
                stagnation_gen = 0
                last_best_val = best_val
            else:
                stagnation_gen += 1

            if stagnation_gen >= stag_limit and evals < budget - ls_budget:
                # Reinitialize worst 25% with Gaussian perturbations around best
                worst_idx = np.argsort(pop_fitness)[-max(1, pop_size // 4):]
                scale = 0.2 * (ub - lb)
                for idx in worst_idx:
                    if evals >= budget - ls_budget:
                        break
                    new_x = best_x + scale * rng.randn(dim)
                    new_x = np.clip(new_x, lb, ub)
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

        # Local search: coordinate pattern search with adaptive step
        remaining = budget - evals
        if remaining > 0:
            best_local = best_x.copy()
            best_local_val = best_val
            step = 0.2 * (ub - lb)
            min_step = 1e-5 * (ub - lb)
            fail_counter = 0
            while evals < budget:
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
                        fail_counter = 0
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
                        fail_counter = 0
                        break
                    else:
                        step[i] = max(step[i] * 0.5, min_step[i])
                if not success:
                    # Random perturbation every 5 failed steps
                    if fail_counter % 5 == 0 and evals < budget:
                        scale = 0.1 * (ub - lb)
                        perturbation = scale * rng.randn(dim)
                        trial = np.clip(best_local + perturbation, lb, ub)
                        val = func(trial)
                        evals += 1
                        if val < best_local_val:
                            best_local_val = val
                            best_local = trial
                            report_best(best_local_val, best_local)
                            step = np.minimum(step * 2, ub - lb)
                            success = True
                            fail_counter = 0
                    if not success:
                        fail_counter += 1
                # If step sizes become too small, restart from a random point
                if evals < budget and np.all(step <= min_step):
                    new_x = lb + rng.rand(dim) * (ub - lb)
                    new_val = func(new_x)
                    evals += 1
                    if new_val < best_local_val:
                        best_local_val = new_val
                        best_local = new_x.copy()
                        report_best(best_local_val, best_local)
                    step = 0.2 * (ub - lb)
                    fail_counter = 0
            # Update global best
            if best_local_val < best_val:
                best_val = best_local_val
                best_x = best_local.copy()

        return best_val, best_x