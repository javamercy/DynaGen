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

        # Population size: small for exploitation
        pop_size = max(4, 2 * dim)
        if pop_size * 2 > budget:
            pop_size = max(4, budget // 2)

        # Allocate 80% of budget to DE, rest to local search
        de_budget = int(budget * 0.8)
        if de_budget < pop_size:
            de_budget = pop_size

        pop = rng.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        evals = 0
        best_x = None
        best_val = np.inf

        # Initial evaluation
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

        # DE parameters: conservative for exploitation
        F = 0.4
        CR = 0.9
        patience_evals = max(1, int(0.1 * de_budget))
        evals_without_improvement = 0

        # Main DE loop (uses de_budget)
        while evals < de_budget:
            for i in range(pop_size):
                if evals >= de_budget:
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
                    # Restart: reinitialize all except best
                    for j in range(pop_size):
                        if evals >= de_budget:
                            break
                        if j != i:  # keep best? Actually keep global best separately
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

        # Local search phase (remaining budget)
        remaining = budget - evals
        if remaining > 0 and best_x is not None:
            step_size = 0.01 * (ub - lb)  # initial step
            for _ in range(remaining):
                if evals >= budget:
                    break
                # Annealed step size: linearly decrease to 0
                frac = evals / budget
                step = step_size * (1 - frac)
                trial = best_x + rng.normal(0, step, dim)
                trial = np.clip(trial, lb, ub)
                trial_val = func(trial)
                evals += 1
                if trial_val < best_val:
                    best_val = trial_val
                    best_x = trial.copy()
                    try:
                        report_best(best_val, best_x)
                    except NameError:
                        pass

        return best_val, best_x