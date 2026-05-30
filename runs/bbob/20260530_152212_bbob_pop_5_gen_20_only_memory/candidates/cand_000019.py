import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        # Smaller population to allocate more budget to local search
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
        # Initial step size for local refinement
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

        # Main loop
        generation = 0
        while evals < budget:
            # DE/best/1/bin mutation (more exploitation)
            F = F0 * (1.0 - evals / budget) ** 0.5
            # Generate offspring
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Select two distinct random indices different from i
                idxs = [j for j in range(pop_size) if j != i]
                a, b = np.random.choice(idxs, 2, replace=False)
                # Mutation: best/1
                mutant = best_x + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                # Crossover: binomial
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                # Evaluation
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # Multiple local refinement steps around best
            if evals < budget:
                # Compute remaining budget ratio
                ratio = 1.0 - evals / budget
                sigma = sigma0 * ratio ** 0.5
                # Number of local steps: proportional to remaining budget
                n_local = min(5, budget - evals)
                for _ in range(n_local):
                    if evals >= budget:
                        break
                    # Random perturbation
                    trial = best_x + np.random.normal(0, sigma, dim)
                    trial = np.clip(trial, lb, ub)
                    val = func(trial)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        # Also replace a random population member to propagate good genes
                        idx = np.random.randint(pop_size)
                        pop[idx] = trial
                        fitness[idx] = val
                    else:
                        # If no improvement, shrink step for next attempts
                        sigma *= 0.9

            generation += 1

        return best_val, best_x