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
        rng = self.rng
        budget = self.budget

        pop_size = min(budget, max(4, min(5 * dim, budget // 3)))

        # Latin Hypercube Sampling
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

        F = 0.5
        CR = 0.9

        sigma = 0.1 * (ub - lb).mean()
        sigma_max = 0.2 * (ub - lb).mean()
        sigma_min = 1e-5 * (ub - lb).mean()
        local_freq = 10
        best_f_unchanged = 0
        stagnation_threshold = max(200, 5 * dim)
        restart_fraction = 0.2

        de_iteration = 0
        while evals < budget:
            # DE step
            target_idx = rng.randint(pop_size)
            candidates = [i for i in range(pop_size) if i != target_idx]
            if len(candidates) < 3:
                continue
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
                    best_f_unchanged = 0
                else:
                    best_f_unchanged += 1
            else:
                best_f_unchanged += 1

            de_iteration += 1

            # Local refinement
            if de_iteration % local_freq == 0 and evals < budget:
                direction = rng.randn(dim)
                norm = np.linalg.norm(direction) + 1e-12
                direction /= norm
                candidate = best_x + sigma * direction
                candidate = np.clip(candidate, lb, ub)
                f_candidate = func(candidate)
                evals += 1
                if f_candidate < best_f:
                    best_f = f_candidate
                    best_x = candidate.copy()
                    report_best(best_f, best_x)
                    sigma = min(sigma * 1.2, sigma_max)
                    best_f_unchanged = 0
                else:
                    sigma = max(sigma * 0.8, sigma_min)
                    best_f_unchanged += 1

            # Restart on stagnation
            if best_f_unchanged >= stagnation_threshold and evals < budget:
                n_restart = int(restart_fraction * pop_size)
                if n_restart > 0:
                    for i in range(n_restart):
                        idx = rng.randint(pop_size)
                        pert = rng.randn(dim) * sigma * 0.1
                        new_x = np.clip(best_x + pert, lb, ub)
                        f_new = func(new_x)
                        evals += 1
                        points[idx] = new_x
                        pop_fitness[idx] = f_new
                        if f_new < best_f:
                            best_f = f_new
                            best_x = new_x.copy()
                            report_best(best_f, best_x)
                best_f_unchanged = 0
                sigma = 0.1 * (ub - lb).mean()

        return best_f, best_x