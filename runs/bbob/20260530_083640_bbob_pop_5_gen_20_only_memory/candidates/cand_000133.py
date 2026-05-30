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

        # Population size
        pop_size = max(4, min(10, budget // 10))
        de_budget = int(0.8 * budget)
        de_budget = max(pop_size, de_budget)

        # Latin hypercube sampling
        intervals = np.linspace(0, 1, pop_size + 1)
        lhs = np.zeros((pop_size, dim))
        for d in range(dim):
            samples = rng.uniform(intervals[:-1], intervals[1:], size=pop_size)
            lhs[:, d] = samples[rng.permutation(pop_size)]
        pop = lb + lhs * (ub - lb)
        pop_fit = np.full(pop_size, np.inf)

        best_val = np.inf
        best_x = None
        evals = 0

        # Evaluate initial population
        for i in range(pop_size):
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            evals += 1
            pop_fit[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            if evals >= budget:
                return best_val, best_x

        # DE phase
        max_gen = (de_budget - evals) // pop_size if de_budget > evals else 0
        stagnation_gen = 0
        stag_limit = max(5, max_gen // 5) if max_gen > 0 else 1
        last_best_val = best_val

        for gen in range(max_gen):
            if evals >= budget:
                break
            frac = gen / max_gen if max_gen > 0 else 0.0
            F = 0.5 + 0.4 * frac  # 0.5->0.9
            CR = 0.9 - 0.4 * frac  # 0.9->0.5
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Select two distinct random indices
                indices = [j for j in range(pop_size) if j != i]
                rng.shuffle(indices)
                a, b = indices[0], indices[1]
                # current-to-best/1
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[a] - pop[b])
                # Jitter
                jitter = 0.001 * rng.randn(dim)
                mutant = mutant + jitter * (ub - lb)
                # Crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                trial = np.clip(trial, lb, ub)
                # Evaluate
                val = func(trial)
                evals += 1
                if val < pop_fit[i]:
                    pop[i] = trial
                    pop_fit[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            # Stagnation detection
            if best_val < last_best_val:
                stagnation_gen = 0
                last_best_val = best_val
            else:
                stagnation_gen += 1
            if stagnation_gen >= stag_limit and evals < budget:
                # Reinitialize worst 25% with Gaussian perturbations around best
                worst_idx = np.argsort(pop_fit)[-max(1, pop_size // 4):]
                scale = 0.2 * (ub - lb)
                for idx in worst_idx:
                    if evals >= budget:
                        break
                    new_x = best_x + scale * rng.randn(dim)
                    new_x = np.clip(new_x, lb, ub)
                    val = func(new_x)
                    evals += 1
                    pop[idx] = new_x
                    pop_fit[idx] = val
                    if val < best_val:
                        best_val = val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)
                stagnation_gen = 0
                last_best_val = best_val

        # Local search: pattern search with random perturbations
        remaining = budget - evals
        if remaining > 0:
            local_x = best_x.copy()
            local_val = best_val
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
                    trial = local_x.copy()
                    trial[i] = np.clip(local_x[i] + step[i], lb[i], ub[i])
                    val = func(trial)
                    evals += 1
                    if val < local_val:
                        local_val = val
                        local_x = trial
                        report_best(local_val, local_x)
                        step[i] = min(step[i] * 2, ub[i] - lb[i])
                        success = True
                        fail_counter = 0
                        break
                    # Negative direction
                    trial[i] = np.clip(local_x[i] - step[i], lb[i], ub[i])
                    val = func(trial)
                    evals += 1
                    if val < local_val:
                        local_val = val
                        local_x = trial
                        report_best(local_val, local_x)
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
                        trial = np.clip(local_x + perturbation, lb, ub)
                        val = func(trial)
                        evals += 1
                        if val < local_val:
                            local_val = val
                            local_x = trial
                            report_best(local_val, local_x)
                            step = np.minimum(step * 2, ub - lb)
                            success = True
                            fail_counter = 0
                    if not success:
                        fail_counter += 1
                # Restart if step sizes too small
                if np.all(step <= min_step) and evals < budget:
                    new_x = lb + rng.rand(dim) * (ub - lb)
                    val = func(new_x)
                    evals += 1
                    if val < local_val:
                        local_val = val
                        local_x = new_x.copy()
                        report_best(local_val, local_x)
                    step = 0.2 * (ub - lb)
                    fail_counter = 0
            # Update global best
            if local_val < best_val:
                best_val = local_val
                best_x = local_x.copy()

        return best_val, best_x