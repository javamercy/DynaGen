import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(3, min(budget // 4, 5 * dim))

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        dim = self.dim
        pop_size = self.pop_size
        F = 0.8
        CR = 0.9

        # Initialization
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0

        # Evaluate initial population
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

        # Compute initial temperature from fitness spread
        if evals >= budget:
            return best_val, best_x
        min_fit = np.min(fitness[:evals])
        max_fit = np.max(fitness[:evals])
        if max_fit - min_fit > 1e-12:
            T0 = (max_fit - min_fit) / 10.0
        else:
            T0 = 1.0

        # Track last improvement
        last_improvement_evals = evals

        # Main loop
        while evals < budget:
            # Update temperature
            ratio = 1.0 - evals / budget
            T = T0 * ratio ** 2

            # DE/best/1/bin with SA acceptance
            best_idx = np.argmin(fitness[:pop_size])
            best = pop[best_idx].copy()
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Choose two distinct random indices != i
                candidates = [j for j in range(pop_size) if j != i]
                a, b = np.random.choice(candidates, 2, replace=False)
                mutant = best + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                val = func(trial)
                evals += 1
                # SA acceptance
                if val < fitness[i] or np.random.rand() < np.exp((fitness[i] - val) / (T + 1e-12)):
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        last_improvement_evals = evals
                # Also update best if trial is better than overall best but not accepted? (above already)
                # Actually the acceptance condition already updates best if better.

            # Local refinement if no improvement for a while
            if evals - last_improvement_evals > max(1, budget // 10) and evals < budget:
                n_local = min(10, budget - evals)
                step = 0.01 * (ub - lb).mean()
                for _ in range(n_local):
                    if evals >= budget:
                        break
                    delta = np.random.normal(0, step, size=dim)
                    trial = best_x + delta
                    trial = np.clip(trial, lb, ub)
                    val = func(trial)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        last_improvement_evals = evals
                        # Replace a random population member
                        idx = np.random.randint(pop_size)
                        pop[idx] = trial
                        fitness[idx] = val
                    step *= 0.9  # reduce step size

        return best_val, best_x