import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        # Larger population for exploration
        self.pop_size = max(5, min(budget // 2, 10 * dim))
        self.F = 0.9
        self.CR = 0.9
        self.stagnation_threshold = max(5, dim)

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        dim = self.dim
        pop_size = self.pop_size
        F = self.F
        CR = self.CR

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
        stagnation = 0
        generation = 0
        while evals < budget:
            # DE/rand/1/bin mutation
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Select three distinct random indices different from i
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                # Mutation: rand/1
                mutant = pop[a] + F * (pop[b] - pop[c])
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

            # Check for improvement
            old_best = best_val
            # After each generation, stagnation counter
            if best_val < old_best - 1e-15:
                stagnation = 0
            else:
                stagnation += 1

            # Restart if stagnation
            if stagnation >= self.stagnation_threshold and evals < budget:
                # Reinitialize all but best
                for i in range(pop_size):
                    if evals >= budget:
                        break
                    # Reinitialize with random point
                    x = np.random.uniform(lb, ub, dim)
                    # Optionally perturb best? Not doing here to keep exploration.
                    pop[i] = x
                    val = func(x)
                    evals += 1
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                # Keep best in population
                if best_x is not None:
                    # Replace worst with best to ensure best is present (though it might already be there)
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_x
                    fitness[worst_idx] = best_val
                stagnation = 0

            generation += 1

        return best_val, best_x