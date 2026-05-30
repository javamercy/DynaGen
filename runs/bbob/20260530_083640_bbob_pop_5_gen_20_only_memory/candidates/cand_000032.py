import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = np.random.RandomState(self.seed)

        # Larger population for exploration
        pop_size = min(15 * dim, budget // 2)
        pop_size = max(pop_size, 10)

        best_val = np.inf
        best_x = None
        evals = 0

        # Initial population
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        pop_vals = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            pop_vals[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        if evals >= budget:
            return best_val, best_x

        # DE parameters for exploration
        F = 0.9
        CR = 0.5
        max_gen = max(1, int(0.6 * budget / pop_size))
        stagnation_counter = 0
        for gen in range(max_gen):
            if evals >= budget:
                break
            improved = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(idxs, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < pop_vals[i]:
                    pop[i] = trial
                    pop_vals[i] = val
                    improved = True
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            if not improved:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            # Restart if stagnation
            if stagnation_counter >= 3:
                # Reinitialize worst half
                worst_indices = np.argsort(pop_vals)[-pop_size//2:]
                for idx in worst_indices:
                    if evals >= budget:
                        break
                    pop[idx] = rng.uniform(lb, ub, size=dim)
                    val = func(pop[idx])
                    evals += 1
                    pop_vals[idx] = val
                    if val < best_val:
                        best_val = val
                        best_x = pop[idx].copy()
                        report_best(best_val, best_x)
                stagnation_counter = 0

        # Final exploratory phase with larger steps
        if evals < budget:
            remaining = budget - evals
            step0 = 0.2 * (ub - lb)
            for i in range(remaining):
                if evals >= budget:
                    break
                alpha = 1.0 - i / remaining
                step = step0 * alpha * (1 + rng.rand())
                x = best_x + rng.randn(dim) * step
                x = np.clip(x, lb, ub)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)

        return best_val, best_x