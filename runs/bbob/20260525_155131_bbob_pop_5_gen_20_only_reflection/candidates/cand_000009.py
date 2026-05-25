import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        span = ub - lb
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # Population size
        NP = max(4, min(budget // (2*dim), 10*dim))
        NP = min(NP, budget // 2)  # ensure enough budget for initial eval
        if NP < 2:
            NP = 2

        pop = lb + rng.rand(NP, dim) * span
        fitness = np.full(NP, np.inf)
        best_val = np.inf
        best_x = np.zeros(dim)
        evals = 0

        # Initial evaluation
        for i in range(NP):
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # Parameters
        step_size = 0.1 * span
        F = 0.8
        CR = 0.9
        success_rate_target = 0.2

        # Main loop
        while evals < budget:
            # Coordinate search on each individual
            successes = 0
            attempts = 0
            order = rng.permutation(NP)
            for i in order:
                if evals >= budget:
                    break
                # Select random subset of coordinates
                n_coords = max(1, min(dim // 4, dim))
                coords = rng.choice(dim, size=n_coords, replace=False)
                for j in coords:
                    if evals >= budget:
                        break
                    step = step_size[j]
                    # Try positive direction
                    x_new = pop[i].copy()
                    x_new[j] = np.clip(pop[i][j] + step, lb[j], ub[j])
                    val_new = func(x_new)
                    evals += 1
                    attempts += 1
                    if val_new < fitness[i]:
                        pop[i] = x_new
                        fitness[i] = val_new
                        if val_new < best_val:
                            best_val = val_new
                            best_x = x_new.copy()
                            report_best(best_val, best_x)
                        successes += 1
                        break  # move to next individual
                    else:
                        # Try negative direction
                        x_new2 = pop[i].copy()
                        x_new2[j] = np.clip(pop[i][j] - step, lb[j], ub[j])
                        val_new2 = func(x_new2)
                        evals += 1
                        attempts += 1
                        if val_new2 < fitness[i]:
                            pop[i] = x_new2
                            fitness[i] = val_new2
                            if val_new2 < best_val:
                                best_val = val_new2
                                best_x = x_new2.copy()
                                report_best(best_val, best_x)
                            successes += 1
                            break
                        # else no change
            # Adapt step size
            if attempts > 0:
                success_rate = successes / attempts
                if success_rate > success_rate_target:
                    step_size = np.clip(step_size * 1.1, 0.01 * span, 0.5 * span)
                else:
                    step_size = np.clip(step_size * 0.9, 0.01 * span, 0.5 * span)

            # Recombination via DE on worst half
            if evals >= budget:
                break
            sort_idx = np.argsort(fitness)
            half = NP // 2
            best_half = pop[sort_idx[:half]]
            worst_half_idx = sort_idx[half:]
            for idx in worst_half_idx:
                if evals >= budget:
                    break
                # Select three distinct parents from best half
                parents = rng.choice(half, size=3, replace=False)
                a, b, c = parents
                mutant = best_half[a] + F * (best_half[b] - best_half[c])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover with a random best half member
                best_parent = best_half[rng.randint(half)]
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, best_parent)
                trial[j_rand] = mutant[j_rand]
                val_trial = func(trial)
                evals += 1
                if val_trial < best_val:
                    best_val = val_trial
                    best_x = trial.copy()
                    report_best(best_val, best_x)
                # Replace worst if better than current
                if val_trial < fitness[idx]:
                    pop[idx] = trial
                    fitness[idx] = val_trial

        return best_val, best_x