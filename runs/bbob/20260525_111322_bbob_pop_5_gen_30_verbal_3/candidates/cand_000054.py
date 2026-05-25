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

        # Population size schedule: start large, decrease linearly to min 4
        pop_start = min(budget, max(10, min(10 * dim, budget // 2)))
        pop_end = 4
        # We'll compute generation count indirectly
        # We'll maintain current population size
        pop_size = pop_start

        # LHS initialization
        points = np.zeros((pop_size, dim))
        for i in range(dim):
            perm = rng.permutation(pop_size)
            u = rng.rand(pop_size)
            points[:, i] = lb[i] + (perm + u) / pop_size * (ub[i] - lb[i])

        pop_fitness = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0

        for i in range(pop_size):
            if evals >= budget:
                break
            x = points[i]
            f = func(x)
            evals += 1
            pop_fitness[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        if evals >= budget:
            return best_f, best_x

        # Schedule parameters: F decreasing, CR increasing
        # We'll compute based on fraction of budget used so far
        def get_schedule(evals, budget):
            frac = evals / budget
            F = 0.9 - 0.7 * frac  # from 0.9 to 0.2
            CR = 0.5 + 0.5 * frac  # from 0.5 to 1.0
            return F, CR

        # Local refinement parameters
        sigma = 0.2 * (ub - lb)
        local_ref_evals = 0  # counter since last local refinement
        # Compute initial local_ref_interval
        def get_local_interval(evals, budget):
            remaining = budget - evals
            if remaining <= 10:
                return 1
            else:
                return max(1, remaining // 10)
        local_interval = get_local_interval(evals, budget)

        while evals < budget:
            # Possibly update population size
            # We'll recompute pop_size based on evaluations used
            new_pop_size = max(pop_end, int(pop_start - (pop_start - pop_end) * (evals / budget)))
            if new_pop_size != pop_size:
                # Adjust population arrays: if new_pop_size smaller, keep best individuals?
                # Simple: keep the first new_pop_size individuals sorted by fitness? But order matters for DE iteration.
                # Instead, we'll keep the existing population; DE operates on first pop_size individuals.
                # We'll just update pop_size variable for future iterations; but we have fixed arrays.
                # To avoid complexity, we'll not resize; we'll just keep the current size.
                # However, the schedule will still affect F and CR.
                pass
            pop_size = new_pop_size  # for reference

            F, CR = get_schedule(evals, budget)

            # DE iteration
            target_idx = rng.randint(pop_size)
            candidates = list(range(pop_size))
            candidates.remove(target_idx)
            if len(candidates) >= 3:
                idx = rng.choice(candidates, 3, replace=False)
                a, b, c = idx
                mutant = points[a] + F * (points[b] - points[c])
                trial = points[target_idx].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                if f_trial < pop_fitness[target_idx]:
                    points[target_idx] = trial
                    pop_fitness[target_idx] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
            else:
                # Not enough distinct, sample random
                if evals >= budget:
                    break
                x = lb + rng.rand(dim) * (ub - lb)
                f = func(x)
                evals += 1
                worst_idx = np.argmax(pop_fitness)
                if f < pop_fitness[worst_idx]:
                    points[worst_idx] = x
                    pop_fitness[worst_idx] = f
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)

            # Local refinement
            if evals >= budget:
                break
            local_ref_evals += 1
            # Update interval based on remaining budget
            local_interval = get_local_interval(evals, budget)
            if local_ref_evals >= local_interval:
                local_ref_evals = 0
                delta = sigma * rng.randn(dim)
                candidate = best_x + delta
                candidate = np.clip(candidate, lb, ub)
                f_candidate = func(candidate)
                evals += 1
                if f_candidate < best_f:
                    best_f = f_candidate
                    best_x = candidate.copy()
                    sigma *= 1.1
                    report_best(best_f, best_x)
                else:
                    sigma *= 0.9

        return best_f, best_x