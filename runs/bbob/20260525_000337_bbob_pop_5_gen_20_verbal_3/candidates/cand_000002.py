import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Population size: at least 10, but not too large relative to budget
        pop_size = max(10, 4 * dim)
        if pop_size * 2 > budget:
            pop_size = max(4, budget // 2)

        # Initialize population uniformly within bounds
        pop = np.random.uniform(lb, ub, (pop_size, dim))

        # Evaluate initial population
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
                # report_best should be called, but it's a global helper; we simulate here
                # In actual environment, report_best(best_val, best_x) is called
                # We will call it when we know the helper exists, but to avoid NameError we comment
                # However the instructions say use report_best, so we must call it.
                # Actually the contract says "A global helper report_best(value, x) is available."
                # So we should call it. We'll assume it's defined elsewhere.
                # But to be safe, we'll include the call in the code.
                try:
                    report_best(best_val, best_x)
                except NameError:
                    pass

        # Main DE loop
        F = 0.5
        CR = 0.9
        generation = 0
        while evals < budget:
            generation += 1
            # Iterate over each individual
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Choose three distinct indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

                # Mutant vector
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                # Binomial crossover
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                # Ensure within bounds
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_val = func(trial)
                evals += 1

                # Greedy selection
                if trial_val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_val
                    if trial_val < best_val:
                        best_val = trial_val
                        best_x = trial.copy()
                        try:
                            report_best(best_val, best_x)
                        except NameError:
                            pass
            # If population stagnates (optional), could restart, but not necessary

        return best_val, best_x