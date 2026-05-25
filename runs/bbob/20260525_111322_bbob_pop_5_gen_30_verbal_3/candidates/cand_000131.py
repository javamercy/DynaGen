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

        pop_size = max(4, min(6 * dim, budget // 3))
        # LHS initialization
        points = np.empty((pop_size, dim))
        for d in range(dim):
            perm = rng.permutation(pop_size)
            u = rng.rand(pop_size)
            points[:, d] = lb[d] + (perm + u) / pop_size * (ub[d] - lb[d])

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

        CR = 0.9
        F = 0.7
        F_min = 0.2
        F_max = 1.0
        gen_success_rates = []
        gen_window = 5

        diversity_threshold = 0.01 * np.mean(ub - lb)
        local_search_freq = 10
        gen_counter = 0

        while evals < budget:
            gen_success = 0
            for _ in range(pop_size):
                if evals >= budget:
                    break
                target_idx = rng.randint(pop_size)
                candidates = [i for i in range(pop_size) if i != target_idx]
                if len(candidates) < 3:
                    continue
                idx = rng.choice(candidates, 3, replace=False)
                a, b, c = idx
                mutant = points[a] + F * (points[b] - points[c])
                trial = points[target_idx].copy()
                j_rand = rng.randint(dim)
                mask = rng.rand(dim) < CR
                mask[j_rand] = True
                trial[mask] = mutant[mask]
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                if f_trial < pop_fitness[target_idx]:
                    points[target_idx] = trial
                    pop_fitness[target_idx] = f_trial
                    gen_success += 1
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)

            # Update F
            if pop_size > 0:
                success_rate = gen_success / pop_size
                gen_success_rates.append(success_rate)
                if len(gen_success_rates) > gen_window:
                    gen_success_rates.pop(0)
                avg_success = np.mean(gen_success_rates) if gen_success_rates else 0.5
                if avg_success > 0.2:
                    F = min(F * 1.05, F_max)
                else:
                    F = max(F * 0.95, F_min)

            gen_counter += 1

            # Diversity check and restart
            if gen_counter % 5 == 0 and evals < budget:
                # Compute mean pairwise distance
                total_dist = 0.0
                count = 0
                for i in range(pop_size):
                    for j in range(i+1, pop_size):
                        total_dist += np.linalg.norm(points[i] - points[j])
                        count += 1
                avg_dist = total_dist / count if count > 0 else 0.0
                if avg_dist < diversity_threshold:
                    # Replace worst half with random points
                    worst_indices = np.argsort(pop_fitness)[-pop_size//2:]
                    for idx in worst_indices:
                        if evals >= budget:
                            break
                        x = lb + rng.rand(dim) * (ub - lb)
                        f = func(x)
                        evals += 1
                        points[idx] = x
                        pop_fitness[idx] = f
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                            report_best(best_f, best_x)

            # Occasional local search (every local_search_freq generations)
            if gen_counter % local_search_freq == 0 and evals < budget:
                sigma = 0.05 * (ub - lb)
                for _ in range(min(2, budget - evals)):
                    delta = sigma * rng.randn(dim)
                    candidate = best_x + delta
                    candidate = np.clip(candidate, lb, ub)
                    f_candidate = func(candidate)
                    evals += 1
                    if f_candidate < best_f:
                        best_f = f_candidate
                        best_x = candidate.copy()
                        sigma *= 1.2
                        report_best(best_f, best_x)
                    else:
                        sigma *= 0.8

        return best_f, best_x