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

        # Latin Hypercube Sampling initialization
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

        DE parameters
        F = 0.5
        CR = 0.9

        Local search parameters
        sigma = 0.1 * (ub - lb).mean()
        sigma_max = 0.2 * (ub - lb).mean()
        sigma_min = 1e-5 * (ub - lb).mean()
        local_freq = 5
        local_success = True  # for adaptation

        Stagnation tracking
        last_improvement_evals = evals  # when best last improved
        restart_threshold = max(10, int(0.05 * budget))  # evals without improvement before restart

        de_iteration = 0
        while evals < budget:
            # DE step
            target_idx = rng.randint(pop_size)
            candidates = list(range(pop_size))
            candidates.remove(target_idx)
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
                    last_improvement_evals = evals

            de_iteration += 1

            # Local search with adaptive frequency
            if de_iteration % local_freq == 0 and evals < budget:
                direction = rng.randn(dim)
                norm = np.linalg.norm(direction)
                if norm > 1e-12:
                    direction /= norm
                else:
                    direction = np.zeros(dim)
                    direction[rng.randint(dim)] = 1.0
                candidate = best_x + sigma * direction
                candidate = np.clip(candidate, lb, ub)
                f_candidate = func(candidate)
                evals += 1
                if f_candidate < best_f:
                    best_f = f_candidate
                    best_x = candidate.copy()
                    report_best(best_f, best_x)
                    sigma = min(sigma * 1.2, sigma_max)
                    local_success = True
                    last_improvement_evals = evals
                else:
                    sigma = max(sigma * 0.8, sigma_min)
                    local_success = False
                # Adapt frequency
                if local_success:
                    local_freq = max(3, int(local_freq * 0.9))
                else:
                    local_freq = min(20, int(local_freq * 1.2))

            # Restart on stagnation
            if evals - last_improvement_evals > restart_threshold and evals < budget:
                # Reinitialize 70% of population (keep best)
                n_keep = max(1, int(0.3 * pop_size))
                new_points = [best_x.copy()]
                new_fitness = [best_f]
                while len(new_points) < pop_size:
                    x = rng.uniform(lb, ub)
                    new_points.append(x)
                    f = func(x)
                    evals += 1
                    new_fitness.append(f)
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                        last_improvement_evals = evals
                    if evals >= budget:
                        break
                if len(new_points) == pop_size:
                    points = np.array(new_points)
                    pop_fitness = np.array(new_fitness)
                # Reset local parameters
                sigma = 0.1 * (ub - lb).mean()
                local_freq = 5
                last_improvement_evals = evals
                # Reset DE iteration counter
                de_iteration = 0

        return best_f, best_x