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

        # Determine population size (ensure at least 4)
        pop_size = max(4, min(5 * dim, budget // 3))

        # LHS initialization
        def lhs_sample(n, dim, lb, ub):
            points = np.zeros((n, dim))
            for i in range(dim):
                perm = rng.permutation(n)
                u = rng.rand(n)
                points[:, i] = lb[i] + (perm + u) / n * (ub[i] - lb[i])
            return points

        # Initial population
        pop = lhs_sample(pop_size, dim, lb, ub)
        pop_fit = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0

        # Evaluate initial population
        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i]
            f = func(x)
            evals += 1
            pop_fit[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        # DE parameters (to be adapted on restart)
        F = 0.5
        CR = 0.9

        # Stagnation detection
        stagnation_limit = max(1, budget // 10)
        no_improve_evals = 0

        while evals < budget:
            # Main DE iteration
            for i in range(pop_size):
                if evals >= budget:
                    break
                target_idx = rng.randint(pop_size)
                candidates = [j for j in range(pop_size) if j != target_idx]
                if len(candidates) < 3:
                    continue
                a, b, c = rng.choice(candidates, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                trial = pop[target_idx].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                if f_trial < pop_fit[target_idx]:
                    pop[target_idx] = trial
                    pop_fit[target_idx] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        no_improve_evals = 0
                    else:
                        no_improve_evals += 1
                else:
                    no_improve_evals += 1

            # Check stagnation and restart if necessary
            if evals < budget and no_improve_evals >= stagnation_limit:
                # Restart: generate new LHS population, retain best
                new_pop = lhs_sample(pop_size, dim, lb, ub)
                # Replace worst point with best if best is better than worst
                if best_f < np.max(pop_fit):
                    worst_idx = np.argmax(pop_fit)
                    new_pop[worst_idx] = best_x.copy()
                pop = new_pop
                # Re-evaluate new pop (except the retained best, which we already have)
                for i in range(pop_size):
                    if evals >= budget:
                        break
                    # Skip if this point is the retained best (already evaluated)
                    # We need to check if it's exactly the best point
                    # To avoid double evaluation, we can compare with best_x
                    # But best_x might be duplicated; simple approach: evaluate all
                    x = pop[i]
                    f = func(x)
                    evals += 1
                    pop_fit[i] = f
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                # Reset stagnation counter
                no_improve_evals = 0
                # Resample F and CR randomly
                F = rng.uniform(0.2, 0.9)
                CR = rng.uniform(0.5, 1.0)

        return best_f, best_x