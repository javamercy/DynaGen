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

        pop_size = max(4, min(5 * dim, budget // 4))
        # Latin Hypercube Sampling
        points = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = rng.permutation(pop_size)
            u = rng.rand(pop_size)
            points[:, d] = lb[d] + (perm + u) / pop_size * (ub[d] - lb[d])

        fitness = np.full(pop_size, np.inf)
        best_f = np.inf
        best_x = None
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

        F_min, F_max = 0.5, 1.0
        CR_min, CR_max = 0.3, 0.9
        no_improve_evals = 0
        local_step = 0.05 * (ub - lb)
        next_local = int(0.05 * budget)

        while evals < budget:
            new_pop = points.copy()
            new_fitness = fitness.copy()
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 3:
                    continue
                a, b, c = rng.choice(candidates, 3, replace=False)
                F = rng.uniform(F_min, F_max)
                CR = rng.uniform(CR_min, CR_max)
                mutant = points[a] + F * (points[b] - points[c])
                trial = points[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                if f_trial < new_fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        no_improve_evals = 0
                    else:
                        no_improve_evals += 1
                else:
                    no_improve_evals += 1

            points = new_pop
            fitness = new_fitness

            # Restart if stagnation
            if no_improve_evals >= 5 * pop_size:
                for i in range(pop_size):
                    if evals >= budget:
                        break
                    if i == 0:
                        points[i] = best_x
                        fitness[i] = best_f
                    else:
                        points[i] = rng.uniform(lb, ub)
                        f = func(points[i])
                        evals += 1
                        fitness[i] = f
                        if f < best_f:
                            best_f = f
                            best_x = points[i].copy()
                            report_best(best_f, best_x)
                no_improve_evals = 0
                local_step = 0.05 * (ub - lb)

            # Local refinement from best
            if evals >= next_local and evals < budget - 3:
                local_evals = min(3, budget - evals)
                for _ in range(local_evals):
                    perturb = rng.normal(0, 1, dim) * local_step
                    trial = np.clip(best_x + perturb, lb, ub)
                    f_trial = func(trial)
                    evals += 1
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        local_step *= 1.2
                    else:
                        local_step *= 0.8
                    local_step = np.clip(local_step, 1e-3 * (ub - lb), 0.5 * (ub - lb))
                next_local = evals + int(0.05 * budget)

        return best_f, best_x