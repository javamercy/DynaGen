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
        F = 0.5
        CR = 0.9

        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        evals = 0
        best_x = None
        best_val = np.inf

        # Initial evaluation
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

        generation = 0
        stagnation = 0
        prev_best = best_val

        while evals < budget:
            # Local refinement if stagnation and budget permits
            if stagnation >= 3 and budget - evals > pop_size:
                # Sample local perturbations around best
                sigma_init = np.sqrt(np.mean((ub - lb)**2)) / 100
                for _ in range(min(pop_size, budget - evals)):
                    sigma = sigma_init / (generation + 1)
                    trial = best_x + np.random.normal(0, sigma, size=dim)
                    trial = np.clip(trial, lb, ub)
                    val = func(trial)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                    if evals >= budget:
                        break
                stagnation = 0

            # DE generation
            best_idx = np.argmin(fitness)
            best = pop[best_idx]
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Choose two distinct random indices different from i and best_idx
                indices = list(range(pop_size))
                if i in indices:
                    indices.remove(i)
                if best_idx in indices:
                    indices.remove(best_idx)
                if len(indices) < 2:
                    continue
                a, b = np.random.choice(indices, 2, replace=False)
                mutant = best + F * (pop[a] - pop[b])
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

            # Update stagnation
            if best_val < prev_best - 1e-12:
                stagnation = 0
                prev_best = best_val
            else:
                stagnation += 1

            generation += 1

        return best_val, best_x