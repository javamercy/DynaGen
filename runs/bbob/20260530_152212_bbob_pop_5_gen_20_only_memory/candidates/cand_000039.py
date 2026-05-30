import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        # larger population for exploration
        self.pop_size = max(10, min(budget // 2, 4 * dim))

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        dim = self.dim
        pop_size = self.pop_size
        F = 0.9  # high mutation for exploration
        CR = 0.9  # high crossover
        evals = 0
        best_x = None
        best_val = np.inf

        # Initialize population uniformly
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
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

        # DE main loop
        stagnation_count = 0
        while evals < budget:
            improved_in_gen = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                # select three distinct random indices
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                # DE/rand/1
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
                j_rand = np.random.randint(0, dim)
                mask = np.random.rand(dim) < CR
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    improved_in_gen = True
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            if improved_in_gen:
                stagnation_count = 0
            else:
                stagnation_count += 1
                # restart if stagnation for many generations
                if stagnation_count >= max(5, 10 * pop_size // dim):
                    # replace worst half with random points, keep best
                    sorted_idx = np.argsort(fitness)
                    num_replace = pop_size // 2
                    for idx in sorted_idx[-num_replace:]:
                        pop[idx] = np.random.uniform(lb, ub, dim)
                        val = func(pop[idx])
                        evals += 1
                        if evals >= budget:
                            break
                        fitness[idx] = val
                        if val < best_val:
                            best_val = val
                            best_x = pop[idx].copy()
                            report_best(best_val, best_x)
                    stagnation_count = 0

        # optional local search via random perturbations, if budget remains
        if evals < budget:
            x = best_x.copy()
            f = best_val
            step = 0.1 * (ub - lb)  # initial step size per dimension
            for _ in range(50):  # max perturbations
                if evals >= budget:
                    break
                # random direction
                direction = np.random.randn(dim)
                direction /= np.linalg.norm(direction) + 1e-12
                # random step size (log-uniform)
                step_size = step * np.exp(np.random.randn() * 0.5)
                trial = x + step_size * direction
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < f:
                    f = val
                    x = trial.copy()
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                    step *= 1.2  # increase step on success
                else:
                    step *= 0.9  # decrease step on failure
                if step < 1e-15:
                    break

        return best_val, best_x