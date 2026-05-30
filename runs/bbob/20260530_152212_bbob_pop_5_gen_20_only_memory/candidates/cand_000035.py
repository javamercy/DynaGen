import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(3, min(budget // 5, 5 * dim))

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        pop_size = self.pop_size
        evals = 0

        # Initialize population
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        if evals >= budget:
            return best_val, best_x

        F0 = 0.8
        CR = 0.9
        stagnation = 0
        step0 = (ub - lb) * 0.1
        step0 = np.maximum(step0, 1e-8)

        while evals < budget:
            # DE generation
            F = F0 * (1.0 - evals / budget) ** 0.5
            for i in range(pop_size):
                if evals >= budget:
                    break
                idxs = list(range(pop_size))
                idxs.remove(i)
                a, b = np.random.choice(idxs, 2, replace=False)
                mutant = best_x + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        stagnation = 0
                    else:
                        stagnation += 1
                else:
                    stagnation += 1

            # Local pattern search from best
            if evals < budget:
                local_budget = min(2 * dim, budget - evals)
                local_evals = 0
                step = step0.copy()
                x = best_x.copy()
                f = best_val
                improved = False
                while local_evals < local_budget and evals < budget:
                    # Coordinate search
                    for coord in np.random.permutation(dim):
                        if local_evals >= local_budget or evals >= budget:
                            break
                        # positive direction
                        x_new = x.copy()
                        x_new[coord] += step[coord]
                        x_new = np.clip(x_new, lb, ub)
                        val = func(x_new)
                        evals += 1
                        local_evals += 1
                        if val < best_val:
                            best_val = val
                            best_x = x_new.copy()
                            report_best(best_val, best_x)
                            x = x_new.copy()
                            f = val
                            improved = True
                            # Continue stepping in same direction
                            while local_evals < local_budget and evals < budget:
                                x_pat = x.copy()
                                x_pat[coord] += step[coord]
                                x_pat = np.clip(x_pat, lb, ub)
                                val = func(x_pat)
                                evals += 1
                                local_evals += 1
                                if val < best_val:
                                    best_val = val
                                    best_x = x_pat.copy()
                                    report_best(best_val, best_x)
                                    x = x_pat.copy()
                                    f = val
                                else:
                                    break
                            break
                        # negative direction
                        x_new = x.copy()
                        x_new[coord] -= step[coord]
                        x_new = np.clip(x_new, lb, ub)
                        val = func(x_new)
                        evals += 1
                        local_evals += 1
                        if val < best_val:
                            best_val = val
                            best_x = x_new.copy()
                            report_best(best_val, best_x)
                            x = x_new.copy()
                            f = val
                            improved = True
                            while local_evals < local_budget and evals < budget:
                                x_pat = x.copy()
                                x_pat[coord] -= step[coord]
                                x_pat = np.clip(x_pat, lb, ub)
                                val = func(x_pat)
                                evals += 1
                                local_evals += 1
                                if val < best_val:
                                    best_val = val
                                    best_x = x_pat.copy()
                                    report_best(best_val, best_x)
                                    x = x_pat.copy()
                                    f = val
                                else:
                                    break
                            break
                    # Random direction with 10% chance
                    if not improved and np.random.rand() < 0.1:
                        direction = np.random.randn(dim)
                        norm = np.linalg.norm(direction)
                        if norm > 0:
                            direction /= norm
                        step_scaled = step.mean() * direction
                        x_new = np.clip(x + step_scaled, lb, ub)
                        val = func(x_new)
                        evals += 1
                        local_evals += 1
                        if val < best_val:
                            best_val = val
                            best_x = x_new.copy()
                            report_best(best_val, best_x)
                            x = x_new.copy()
                            f = val
                            improved = True
                    if not improved:
                        step *= 0.5
                        stagnation_local = 0
                        # Local restart if stuck
                        if local_evals >= 2 * dim:
                            x = np.clip(best_x + np.random.randn(dim) * (ub - lb) * 0.02, lb, ub)
                            val = func(x)
                            evals += 1
                            local_evals += 1
                            if val < best_val:
                                best_val = val
                                best_x = x.copy()
                                report_best(best_val, best_x)
                                f = val
                            step = step0.copy()
                    else:
                        # Reset step after improvement?
                        step = step0.copy()
                # Replace worst population member with best
                if best_val < fitness[-1]:
                    pop[-1] = best_x.copy()
                    fitness[-1] = best_val

            # Global restart if stagnation too high
            if stagnation > 2 * dim and evals < budget:
                # Reinitialize a point near best
                x_new = np.clip(best_x + np.random.randn(dim) * (ub - lb) * 0.02, lb, ub)
                val = func(x_new)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x_new.copy()
                    report_best(best_val, best_x)
                # Replace worst population member
                worst_idx = np.argmax(fitness)
                pop[worst_idx] = x_new
                fitness[worst_idx] = val
                stagnation = 0

        return best_val, best_x