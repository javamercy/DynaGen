import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng

        pop_size = min(20, max(4, budget // 2))
        if budget <= 0:
            best_x = np.zeros(dim)
            best_f = float('inf')
            report_best(best_f, best_x)
            return best_f, best_x

        # Initialization
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if budget <= 0:
                break
            pop_f[i] = func(pop[i])
            budget -= 1
        evals = min(pop_size, pop_size)  # effectively pop_size
        # if budget ran out early
        if budget <= 0:
            best_idx = np.argmin(pop_f[:evals])
            best_x = pop[best_idx].copy()
            best_f = pop_f[best_idx]
            report_best(best_f, best_x)
            return best_f, best_x

        best_idx = np.argmin(pop_f[:evals])
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        stagnation_limit = max(1, budget // (2 * pop_size))
        stagnation_counter = 0
        generation = 0
        local_search_freq = max(1, budget // 10)
        next_local_search = local_search_freq

        while budget > 0:
            improved = False
            for i in range(pop_size):
                if budget <= 0:
                    break
                # Select two distinct indices
                indices = [j for j in range(pop_size) if j != i]
                r1, r2 = rng.choice(indices, 2, replace=False)
                F = rng.uniform(0.5, 1.0)
                CR = rng.uniform(0.5, 0.9)
                mut = pop[i] + F * (pop[r1] - pop[r2]) + F * (best_x - pop[i])
                # Binomial crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mut[j]
                trial = np.clip(trial, lb, ub)
                trial_f = func(trial)
                budget -= 1
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_x = trial.copy()
                        best_f = trial_f
                        report_best(best_f, best_x)
                        improved = True
            # Local search on best
            if budget > 0 and (self.budget - budget) >= next_local_search:
                next_local_search += local_search_freq
                step = 0.01 * (ub - lb) * rng.randn(dim)
                candidate = np.clip(best_x + step, lb, ub)
                f_candidate = func(candidate)
                budget -= 1
                if f_candidate < best_f:
                    best_x = candidate.copy()
                    best_f = f_candidate
                    report_best(best_f, best_x)
                    # Optionally replace worst individual
                    worst_idx = np.argmax(pop_f)
                    if f_candidate < pop_f[worst_idx]:
                        pop[worst_idx] = candidate
                        pop_f[worst_idx] = f_candidate
            # Stagnation check and restart
            if not improved:
                stagnation_counter += 1
                if stagnation_counter >= stagnation_limit and budget >= pop_size:
                    # Keep best, replace worst half with random
                    sorted_indices = np.argsort(pop_f)
                    keep_idx = sorted_indices[:pop_size // 2]
                    renew_idx = sorted_indices[pop_size // 2:]
                    new_pop = rng.uniform(lb, ub, size=(len(renew_idx), dim))
                    for k, idx in enumerate(renew_idx):
                        if budget <= 0:
                            break
                        pop[idx] = new_pop[k]
                        pop_f[idx] = func(pop[idx])
                        budget -= 1
                    stagnation_counter = 0
            else:
                stagnation_counter = 0

        return best_f, best_x