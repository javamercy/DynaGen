import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(3, min(budget // 5, 4 * dim))

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        dim = self.dim
        pop_size = self.pop_size
        F0 = 0.6
        CR = 0.4
        sigma0 = 0.15 * (ub - lb).mean()

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

        # Main loop
        while evals < budget:
            # DE/current-to-best/1 with low CR for exploitation
            F = F0 * (1.0 - evals / budget) ** 0.3
            for i in range(pop_size):
                if evals >= budget:
                    break
                idxs = [j for j in range(pop_size) if j != i]
                a, b = np.random.choice(idxs, 2, replace=False)
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # Local refinement using elite covariance
            if evals < budget:
                ratio = 1.0 - evals / budget
                sigma = sigma0 * ratio ** 1.5
                n_local = min(20, budget - evals)
                # Compute covariance from elite (best half)
                elite_size = max(2, pop_size // 2)
                sorted_idx = np.argsort(fitness)
                elite = pop[sorted_idx[:elite_size]]
                if elite_size > 1:
                    C = np.cov(elite.T) + 1e-8 * np.eye(dim)
                else:
                    C = np.eye(dim)
                for _ in range(n_local):
                    if evals >= budget:
                        break
                    delta = np.random.multivariate_normal(np.zeros(dim), sigma ** 2 * C)
                    trial = best_x + delta
                    trial = np.clip(trial, lb, ub)
                    val = func(trial)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        # Replace a random population member
                        idx = np.random.randint(pop_size)
                        pop[idx] = trial
                        fitness[idx] = val
                    else:
                        sigma *= 0.85
        return best_val, best_x