import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        pop_size = max(10, 4 * dim)
        if pop_size * 2 > budget:
            pop_size = max(4, budget // 2)

        pop = rng.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        evals = 0
        best_x = None
        best_val = np.inf

        for i in range(pop_size):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_val:
                best_val = fitness[i]
                best_x = pop[i].copy()
                try:
                    report_best(best_val, best_x)
                except NameError:
                    pass

        F = 0.5
        CR = 0.9
        patience_evals = max(1, int(0.1 * budget))
        evals_without_improvement = 0

        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                trial_val = func(trial)
                evals += 1
                if trial_val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_val
                    if trial_val < best_val:
                        best_val = trial_val
                        best_x = trial.copy()
                        evals_without_improvement = 0
                        try:
                            report_best(best_val, best_x)
                        except NameError:
                            pass
                else:
                    evals_without_improvement += 1
                if evals_without_improvement >= patience_evals:
                    # restart: reinitialize all except best
                    for j in range(pop_size):
                        if evals >= budget:
                            break
                        if j != i:  # keep current best? Actually best might be different. Keep global best.
                            pop[j] = rng.uniform(lb, ub, dim)
                            fitness[j] = func(pop[j])
                            evals += 1
                            if fitness[j] < best_val:
                                best_val = fitness[j]
                                best_x = pop[j].copy()
                                evals_without_improvement = 0
                                try:
                                    report_best(best_val, best_x)
                                except NameError:
                                    pass
                    evals_without_improvement = 0
        return best_val, best_x