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
        F0 = 0.8
        CR = 0.9
        sigma0 = 0.2 * (ub - lb).mean()

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

        # Parameters for local refinement
        local_step = sigma0
        success_history = []
        stagnation_counter = 0
        max_stagnation = 5 * pop_size

        while evals < budget:
            # DE/best/1/bin
            F = F0 * (1.0 - evals / budget) ** 0.5
            for i in range(pop_size):
                if evals >= budget:
                    break
                idxs = [j for j in range(pop_size) if j != i]
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
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1

            # Local refinement with covariance and step adaptation
            if evals < budget:
                # Determine number of local steps (use 20% of remaining budget)
                remaining = budget - evals
                n_local = min(10, max(1, remaining // 5))
                # Compute population covariance
                if pop_size > 1:
                    center = pop.mean(axis=0)
                    C = np.cov(pop.T) + 1e-8 * np.eye(dim)
                else:
                    C = np.eye(dim)
                for _ in range(n_local):
                    if evals >= budget:
                        break
                    delta = np.random.multivariate_normal(np.zeros(dim), local_step ** 2 * C)
                    trial = best_x + delta
                    trial = np.clip(trial, lb, ub)
                    val = func(trial)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        local_step *= 1.1  # expand on success
                        success_history.append(1)
                        # Replace worst population member
                        worst_idx = np.argmax(fitness)
                        pop[worst_idx] = trial
                        fitness[worst_idx] = val
                        stagnation_counter = 0
                    else:
                        local_step *= 0.9  # shrink on failure
                        success_history.append(0)
                        if len(success_history) > 10:
                            success_history.pop(0)

            # Restart if stagnation
            if stagnation_counter >= max_stagnation and evals < budget:
                # Reinitialize half of population around best
                new_sigma = sigma0 * 0.1
                for i in range(pop_size // 2):
                    if evals >= budget:
                        break
                    trial = best_x + np.random.randn(dim) * new_sigma
                    trial = np.clip(trial, lb, ub)
                    val = func(trial)
                    evals += 1
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                local_step = new_sigma
                stagnation_counter = 0

        return best_val, best_x