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
        lb, ub = func.bounds.lb, func.bounds.ub
        rng = self.rng

        pop_size = min(5 * dim, max(4, budget // 2))
        if pop_size < 4:
            best_x = rng.uniform(lb, ub, dim)
            best_val = func(best_x)
            report_best(best_val, best_x)
            fcalls = 1
            while fcalls < budget:
                x = rng.uniform(lb, ub, dim)
                val = func(x)
                fcalls += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        pop = rng.uniform(lb, ub, (pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        fcalls = 0

        for i in range(pop_size):
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            fcalls += 1
            pop_f[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        step = 0.1 * (ub - lb)
        min_step = 1e-8 * (ub - lb)
        F = 0.8
        CR = 0.9
        local_search_freq = 2
        generation = 0

        while fcalls < budget:
            generation += 1
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                r0, r1, r2 = rng.choice(candidates, 3, replace=False)
                mutant = pop[r0] + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                trial = pop[i].copy()
                j_rand = rng.randint(0, dim-1)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                fcalls += 1
                if val < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            if fcalls < budget and generation % local_search_freq == 0:
                x_prev = best_x.copy()
                improved = False
                # one cycle of coordinate steps
                for i in range(dim):
                    if fcalls >= budget:
                        break
                    # positive step
                    x_new = best_x.copy()
                    x_new[i] += step[i]
                    x_new[i] = np.clip(x_new[i], lb[i], ub[i])
                    if fcalls < budget:
                        val_new = func(x_new)
                        fcalls += 1
                        if val_new < best_val:
                            best_val = val_new
                            best_x = x_new.copy()
                            report_best(best_val, best_x)
                            step[i] *= 1.2
                            improved = True
                            continue
                    # negative step
                    x_new = best_x.copy()
                    x_new[i] -= step[i]
                    x_new[i] = np.clip(x_new[i], lb[i], ub[i])
                    if fcalls < budget:
                        val_new = func(x_new)
                        fcalls += 1
                        if val_new < best_val:
                            best_val = val_new
                            best_x = x_new.copy()
                            report_best(best_val, best_x)
                            step[i] *= 1.2
                            improved = True
                            continue
                    step[i] *= 0.9
                    if step[i] < min_step[i]:
                        step[i] = min_step[i]

                if improved and fcalls < budget:
                    direction = best_x - x_prev
                    if np.linalg.norm(direction) > 0:
                        factor = 1.0
                        x_pattern = best_x + factor * direction
                        x_pattern = np.clip(x_pattern, lb, ub)
                        if fcalls < budget:
                            val_pattern = func(x_pattern)
                            fcalls += 1
                            if val_pattern < best_val:
                                best_val = val_pattern
                                best_x = x_pattern.copy()
                                report_best(best_val, best_x)

        return best_val, best_x