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
        pop_size = max(4, min(6 * dim, budget // 3))

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

        # DE parameters
        CR = 0.9
        # Local search parameters
        sigma_small = 0.2 * (ub - lb)
        sigma_large = 0.5 * (ub - lb)
        local_ref_interval = max(1, pop_size)
        gen_evals = 0
        stagnation_counter = 0
        max_stagnation = max(10, budget // 15)  # more sensitive stagnation
        diversity_interval = max(1, budget // 5)  # global restart interval
        evals_since_diversity = 0

        while evals < budget:
            # DE iteration: choose strategy randomly
            if rng.rand() < 0.8:
                # DE/rand/1/bin
                target_idx = rng.randint(pop_size)
                candidates = list(range(pop_size))
                candidates.remove(target_idx)
                if len(candidates) >= 3:
                    idx = rng.choice(candidates, 3, replace=False)
                    a, b, c = idx
                    F = 0.5 + rng.rand() * 0.5
                    mutant = points[a] + F * (points[b] - points[c])
                    trial = points[target_idx].copy()
                    j_rand = rng.randint(dim)
                    for j in range(dim):
                        if rng.rand() < CR or j == j_rand:
                            trial[j] = mutant[j]
                    trial = np.clip(trial, lb, ub)
                    f_trial = func(trial)
                    evals += 1
                    gen_evals += 1
                    evals_since_diversity += 1
                    if f_trial < pop_fitness[target_idx]:
                        points[target_idx] = trial
                        pop_fitness[target_idx] = f_trial
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = trial.copy()
                            report_best(best_f, best_x)
                            stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    # fallback: random point
                    if evals >= budget:
                        break
                    x = lb + rng.rand(dim) * (ub - lb)
                    f = func(x)
                    evals += 1
                    gen_evals += 1
                    evals_since_diversity += 1
                    worst_idx = np.argmax(pop_fitness)
                    if f < pop_fitness[worst_idx]:
                        points[worst_idx] = x
                        pop_fitness[worst_idx] = f
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                            report_best(best_f, best_x)
                            stagnation_counter = 0
                    else:
                        stagnation_counter += 1
            else:
                # DE/best/1/bin
                target_idx = rng.randint(pop_size)
                candidates = list(range(pop_size))
                candidates.remove(target_idx)
                if len(candidates) >= 2:
                    idx = rng.choice(candidates, 2, replace=False)
                    b, c = idx
                    F = 0.5 + rng.rand() * 0.5
                    mutant = best_x + F * (points[b] - points[c])
                    trial = points[target_idx].copy()
                    j_rand = rng.randint(dim)
                    for j in range(dim):
                        if rng.rand() < CR or j == j_rand:
                            trial[j] = mutant[j]
                    trial = np.clip(trial, lb, ub)
                    f_trial = func(trial)
                    evals += 1
                    gen_evals += 1
                    evals_since_diversity += 1
                    if f_trial < pop_fitness[target_idx]:
                        points[target_idx] = trial
                        pop_fitness[target_idx] = f_trial
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = trial.copy()
                            report_best(best_f, best_x)
                            stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    # fallback: random point
                    if evals >= budget:
                        break
                    x = lb + rng.rand(dim) * (ub - lb)
                    f = func(x)
                    evals += 1
                    gen_evals += 1
                    evals_since_diversity += 1
                    worst_idx = np.argmax(pop_fitness)
                    if f < pop_fitness[worst_idx]:
                        points[worst_idx] = x
                        pop_fitness[worst_idx] = f
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                            report_best(best_f, best_x)
                            stagnation_counter = 0
                    else:
                        stagnation_counter += 1

            # Local refinement (every local_ref_interval DE generations)
            if gen_evals >= local_ref_interval and evals < budget:
                gen_evals = 0
                # random choice of step size
                if rng.rand() < 0.5:
                    sigma = sigma_small.copy()
                else:
                    sigma = sigma_large.copy()
                delta = sigma * rng.randn(dim)
                candidate = best_x + delta
                candidate = np.clip(candidate, lb, ub)
                f_candidate = func(candidate)
                evals += 1
                evals_since_diversity += 1
                if f_candidate < best_f:
                    best_f = f_candidate
                    best_x = candidate.copy()
                    report_best(best_f, best_x)
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

            # Diversity injection (every diversity_interval evaluations)
            if evals_since_diversity >= diversity_interval and evals < budget:
                evals_since_diversity = 0
                # Keep best, reinitialize 20% of population with LHS
                num_new = max(1, pop_size // 5)
                new_points = np.zeros((num_new, dim))
                for i in range(dim):
                    perm = rng.permutation(num_new)
                    u = rng.rand(num_new)
                    new_points[:, i] = lb[i] + (perm + u) / num_new * (ub[i] - lb[i])
                for i in range(num_new):
                    if evals >= budget:
                        break
                    x = new_points[i]
                    f = func(x)
                    evals += 1
                    # Replace worst in population
                    worst_idx = np.argmax(pop_fitness)
                    if f < pop_fitness[worst_idx]:
                        points[worst_idx] = x
                        pop_fitness[worst_idx] = f
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                            report_best(best_f, best_x)
                            stagnation_counter = 0
                    else:
                        stagnation_counter += 1

            # Restart on stagnation (reinitialize entire population except best)
            if stagnation_counter >= max_stagnation and evals < budget:
                # Keep best, reinitialize rest with LHS
                new_pop_size = pop_size - 1
                new_points = np.zeros((new_pop_size, dim))
                for i in range(dim):
                    perm = rng.permutation(new_pop_size)
                    u = rng.rand(new_pop_size)
                    new_points[:, i] = lb[i] + (perm + u) / new_pop_size * (ub[i] - lb[i])
                # Evaluate and replace all except best's slot
                # first, put best in population as first element to keep it?
                # Simpler: assign best to a specific index, then replace others
                points[0] = best_x.copy()
                pop_fitness[0] = best_f
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    idx = i
                    if idx >= len(points):
                        break
                    x = new_points[idx-1] if idx-1 < new_points.shape[0] else (lb + rng.rand(dim) * (ub - lb))
                    f = func(x)
                    evals += 1
                    points[idx] = x
                    pop_fitness[idx] = f
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                # Ensure best is still in population
                points[0] = best_x.copy()
                pop_fitness[0] = best_f
                stagnation_counter = 0
                evals_since_diversity = 0

        return best_f, best_x