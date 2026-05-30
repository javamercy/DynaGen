import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(4, min(budget // 2, 5 * dim))
        np.random.seed(seed)

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        pop_size = self.pop_size
        F = 0.8
        CR = 0.9

        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        evals = 0
        best_x = None
        best_val = np.inf

        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        step = (ub - lb) * 0.1
        step = np.maximum(step, 1e-6)
        stagnation = 0
        last_improvement_evals = evals

        while evals < budget:
            # check stagnation
            if evals - last_improvement_evals > 2 * dim:
                # reinitialize worst half
                worst_indices = np.argsort(fitness)[-pop_size//2:]
                for idx in worst_indices:
                    if evals >= budget:
                        break
                    pop[idx] = np.random.uniform(lb, ub, dim)
                    val = func(pop[idx])
                    evals += 1
                    fitness[idx] = val
                    if val < best_val:
                        best_val = val
                        best_x = pop[idx].copy()
                        report_best(best_val, best_x)
                        last_improvement_evals = evals
                step = (ub - lb) * 0.1
                step = np.maximum(step, 1e-6)
                stagnation = 0
                continue

            improved_in_generation = False
            # shuffle indices
            indices = np.random.permutation(pop_size)
            for i in indices:
                if evals >= budget:
                    break
                # select indices for mutation
                idxs = [j for j in range(pop_size) if j != i and j != np.argmin(fitness)]
                if len(idxs) < 2:
                    continue
                a, b = np.random.choice(idxs, 2, replace=False)
                best = pop[np.argmin(fitness)]
                mutant = best + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)

                j_rand = np.random.randint(0, dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]

                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    old_x = pop[i].copy()
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        last_improvement_evals = evals
                    # pattern step in direction of improvement
                    if evals < budget:
                        dir = trial - old_x
                        norm = np.linalg.norm(dir)
                        if norm > 1e-12:
                            dir_unit = dir / norm
                            new = np.clip(trial + step * dir_unit, lb, ub)
                            val2 = func(new)
                            evals += 1
                            if val2 < best_val:
                                best_val = val2
                                best_x = new.copy()
                                report_best(best_val, best_x)
                                pop[i] = new
                                fitness[i] = val2
                                last_improvement_evals = evals
                            # randomize step after success
                            step *= np.random.uniform(0.9, 1.1, size=dim)
                    improved_in_generation = True
                else:
                    # reduce step on failure (when no improvement in this trial)
                    step *= 0.99

            if not improved_in_generation:
                stagnation += 1
                step *= 0.5
                step = np.maximum(step, 1e-12)
            else:
                stagnation = 0

        return best_val, best_x