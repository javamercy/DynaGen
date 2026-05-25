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
        pop_size = max(4, min(5 * dim, budget // 3))

        # Latin Hypercube Sampling initialization
        points = np.zeros((pop_size, dim))
        for i in range(dim):
            perm = rng.permutation(pop_size)
            u = rng.rand(pop_size)
            points[:, i] = lb[i] + (perm + u) / pop_size * (ub[i] - lb[i])
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0

        for i in range(pop_size):
            if evals >= budget:
                break
            x = points[i]
            f = func(x)
            evals += 1
            fitness[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        # DE parameters
        F = 0.5
        CR = 0.9
        # Success-based adaptation memory
        F_memory = [F] * 5
        CR_memory = [CR] * 5
        F_idx = 0
        CR_idx = 0
        F_success = []
        CR_success = []

        # Local refinement parameters
        sigma = 0.2 * (ub - lb)
        local_interval = max(1, pop_size)
        gen_evals = 0
        stagnation = 0
        max_stagnation = max(10, budget // 10)
        success_rate = 0.5
        sigma_inc = 1.2
        sigma_dec = 0.9

        while evals < budget:
            target_idx = rng.randint(pop_size)
            candidates = [i for i in range(pop_size) if i != target_idx]
            if len(candidates) < 3:
                # fallback
                if evals >= budget:
                    break
                x = lb + rng.rand(dim) * (ub - lb)
                f = func(x)
                evals += 1
                gen_evals += 1
                worst = np.argmax(fitness)
                if f < fitness[worst]:
                    points[worst] = x
                    fitness[worst] = f
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                        stagnation = 0
                else:
                    stagnation += 1
            else:
                a, b, c = rng.choice(candidates, 3, replace=False)
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
                if f_trial < fitness[target_idx]:
                    points[target_idx] = trial
                    fitness[target_idx] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        stagnation = 0
                    F_success.append(F)
                    CR_success.append(CR)
                else:
                    stagnation += 1

            # Update F and CR based on success history (every 20 generations)
            if evals % (pop_size * 20) == 0 and len(F_success) > 5:
                F = np.mean(F_success)
                CR = np.mean(CR_success)
                F_success = []
                CR_success = []

            # Local refinement
            if gen_evals >= local_interval and evals < budget:
                gen_evals = 0
                delta = sigma * rng.randn(dim)
                candidate = best_x + delta
                candidate = np.clip(candidate, lb, ub)
                f_candidate = func(candidate)
                evals += 1
                if f_candidate < best_f:
                    best_f = f_candidate
                    best_x = candidate.copy()
                    report_best(best_f, best_x)
                    sigma *= sigma_inc
                    stagnation = 0
                    success_rate = 0.9 * success_rate + 0.1
                else:
                    sigma *= sigma_dec
                    success_rate = 0.9 * success_rate
                # Adjust sigma further based on success rate
                if success_rate > 0.2:
                    sigma *= 1.0
                else:
                    sigma *= 0.95

            # Restart on stagnation
            if stagnation >= max_stagnation and evals < budget:
                new_pop_size = max(4, pop_size - 1)
                new_points = np.zeros((new_pop_size, dim))
                for i in range(dim):
                    perm = rng.permutation(new_pop_size)
                    u = rng.rand(new_pop_size)
                    new_points[:, i] = lb[i] + (perm + u) / new_pop_size * (ub[i] - lb[i])
                for i in range(new_pop_size):
                    if evals >= budget:
                        break
                    x = new_points[i]
                    f = func(x)
                    evals += 1
                    worst = np.argmax(fitness)
                    if f < fitness[worst]:
                        points[worst] = x
                        fitness[worst] = f
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                            report_best(best_f, best_x)
                # Replace worst with best
                worst = np.argmax(fitness)
                points[worst] = best_x.copy()
                fitness[worst] = best_f
                stagnation = 0
                sigma = 0.2 * (ub - lb)  # reset sigma
                success_rate = 0.5

        return best_f, best_x