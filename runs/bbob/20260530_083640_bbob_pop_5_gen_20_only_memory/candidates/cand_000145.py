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
        best_x = rng.uniform(lb, ub)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        # Phase 1: Differential Evolution (80% budget)
        de_budget = int(0.8 * budget)
        if de_budget > 1:
            NP = max(3, min(15, dim * 2))
            if NP < 3:
                NP = 3
            if de_budget > NP:
                pop = rng.uniform(lb, ub, size=(NP, dim))
                fitness = np.full(NP, np.inf)
                for i in range(NP):
                    if evals >= de_budget:
                        break
                    val = func(pop[i])
                    evals += 1
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = pop[i].copy()
                        report_best(best_val, best_x)
                F = 0.9
                CR = 0.95
                max_generations = (de_budget - evals) // NP
                for gen in range(max_generations):
                    if evals >= de_budget:
                        break
                    for i in range(NP):
                        if evals >= de_budget:
                            break
                        indices = list(range(NP))
                        indices.remove(i)
                        rng.shuffle(indices)
                        a, b, c = indices[0], indices[1], indices[2]
                        mutant = pop[a] + F * (pop[b] - pop[c])
                        trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                        j_rand = rng.randint(dim)
                        trial[j_rand] = mutant[j_rand]
                        trial = np.clip(trial, lb, ub)
                        val = func(trial)
                        evals += 1
                        if val < fitness[i]:
                            pop[i] = trial
                            fitness[i] = val
                            if val < best_val:
                                best_val = val
                                best_x = trial.copy()
                                report_best(best_val, best_x)

        # Phase 2: Coordinate Search with adaptive steps
        step = 0.2 * (ub - lb)
        max_failures = max(dim * 2, 20)
        failure_counter = 0
        while evals < budget:
            success = False
            perm = rng.permutation(dim)
            for i in perm:
                if evals >= budget:
                    break
                # Positive direction
                trial = best_x.copy()
                trial[i] = np.clip(best_x[i] + step[i], lb[i], ub[i])
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    report_best(best_val, best_x)
                    step[i] = min(step[i] * 2, ub[i] - lb[i])
                    success = True
                    break
                # Negative direction
                trial = best_x.copy()
                trial[i] = np.clip(best_x[i] - step[i], lb[i], ub[i])
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    report_best(best_val, best_x)
                    step[i] = min(step[i] * 2, ub[i] - lb[i])
                    success = True
                    break
                else:
                    step[i] = max(step[i] * 0.5, (ub[i] - lb[i]) * 1e-10)

            # Random direction poll if no success
            if not success and evals < budget:
                direction = rng.randn(dim)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                trial = np.clip(best_x + 2 * step * direction, lb, ub)
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    report_best(best_val, best_x)
                    step = np.minimum(step * 2, ub - lb)
                    success = True

            # Update failure counter
            if success:
                failure_counter = 0
            else:
                failure_counter += 1

            # Random exploration with small probability
            if evals < budget and rng.rand() < 0.05:
                trial = rng.uniform(lb, ub)
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    report_best(best_val, best_x)
                    step = 0.2 * (ub - lb)
                    failure_counter = 0

            # Restart if stagnation
            if failure_counter >= max_failures and evals < budget:
                trial = rng.uniform(lb, ub)
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    report_best(best_val, best_x)
                step = 0.2 * (ub - lb)
                failure_counter = 0

        return best_val, best_x