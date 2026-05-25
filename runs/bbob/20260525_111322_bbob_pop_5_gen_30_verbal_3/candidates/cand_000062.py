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

        pop_size = max(4, min(4 * dim, budget // 4))
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

        # Evaluate initial population
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
        F = 0.5
        F_min = 0.1
        F_max = 1.0
        gen_success_rates = []
        gen_window = 5

        # Local refinement parameters
        sigma = 0.2 * (ub - lb)
        sigma_min = 1e-8 * (ub - lb)
        local_ref_interval = pop_size
        gen_evals = 0

        # Stagnation and restart
        stagnation_counter = 0
        max_stagnation = max(20, budget // 20)

        # Main loop
        while evals < budget:
            gen_success = 0
            for _ in range(pop_size):
                if evals >= budget:
                    break
                # Selection
                target_idx = rng.randint(pop_size)
                candidates = [i for i in range(pop_size) if i != target_idx]
                if len(candidates) < 3:
                    # Fallback: random point
                    x_rand = lb + rng.rand(dim) * (ub - lb)
                    f_rand = func(x_rand)
                    evals += 1
                    worst_idx = np.argmax(pop_fitness)
                    if f_rand < pop_fitness[worst_idx]:
                        points[worst_idx] = x_rand
                        pop_fitness[worst_idx] = f_rand
                        if f_rand < best_f:
                            best_f = f_rand
                            best_x = x_rand.copy()
                            report_best(best_f, best_x)
                            stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                    continue

                idx = rng.choice(candidates, 3, replace=False)
                a, b, c = idx
                mutant = points[a] + F * (points[b] - points[c])
                # Crossover
                trial = points[target_idx].copy()
                j_rand = rng.randint(dim)
                mask = rng.rand(dim) < CR
                mask[j_rand] = True
                trial[mask] = mutant[mask]
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1
                gen_evals += 1
                if f_trial < pop_fitness[target_idx]:
                    points[target_idx] = trial
                    pop_fitness[target_idx] = f_trial
                    gen_success += 1
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        stagnation_counter = 0
                else:
                    stagnation_counter += 1

                # Local refinement after each generation
                if gen_evals >= local_ref_interval and evals < budget:
                    gen_evals = 0
                    delta = sigma * rng.randn(dim)
                    candidate = best_x + delta
                    candidate = np.clip(candidate, lb, ub)
                    f_candidate = func(candidate)
                    evals += 1
                    if f_candidate < best_f:
                        best_f = f_candidate
                        best_x = candidate.copy()
                        sigma = np.clip(sigma * 1.1, None, ub - lb)
                        report_best(best_f, best_x)
                        stagnation_counter = 0
                    else:
                        sigma = np.clip(sigma * 0.9, sigma_min, None)
                        stagnation_counter += 1

            # Update F based on success rate
            if pop_size > 0:
                success_rate = gen_success / pop_size
                gen_success_rates.append(success_rate)
                if len(gen_success_rates) > gen_window:
                    gen_success_rates.pop(0)
                avg_success = np.mean(gen_success_rates) if gen_success_rates else 0.5
                if avg_success > 0.2:
                    F = min(F * 1.1, F_max)
                else:
                    F = max(F * 0.9, F_min)

            # Restart if stagnation
            if stagnation_counter >= max_stagnation and evals < budget:
                new_pop_size = pop_size - 1
                new_points = np.empty((new_pop_size, dim))
                for d in range(dim):
                    perm = rng.permutation(new_pop_size)
                    u = rng.rand(new_pop_size)
                    new_points[:, d] = lb[d] + (perm + u) / new_pop_size * (ub[d] - lb[d])
                for i in range(new_pop_size):
                    if evals >= budget:
                        break
                    x = new_points[i]
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
                worst_idx = np.argmax(pop_fitness)
                points[worst_idx] = best_x.copy()
                pop_fitness[worst_idx] = best_f
                stagnation_counter = 0
                sigma = 0.2 * (ub - lb)

        return best_f, best_x