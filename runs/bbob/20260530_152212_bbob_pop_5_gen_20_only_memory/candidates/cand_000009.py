import numpy as np
class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(3, min(budget // 2, 10 * dim))

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

        # Initialize population
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        evals = 0
        best_x = None
        best_val = np.inf

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
            # Adaptive mutation factor
            F = F0 * (1 - evals / budget) ** 0.5
            # Local refinement step (every generation)
            if generation > 0 and evals < budget:
                sigma = sigma0 * (1 - evals / budget) ** 0.5
                trial = best_x + np.random.normal(0, sigma, size=dim)
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    report_best(best_val, best_x)
                    # Replace a random population member to propagate good genes
                    idx = np.random.randint(pop_size)
                    pop[idx] = trial
                    fitness[idx] = val
            # DE generation
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Mutation: rand/1
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # Crossover: binomial
                j_rand = np.random.randint(0, dim)
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
            generation += 1
        return best_val, best_x