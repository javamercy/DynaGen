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

        # Population size: typical DE setting
        pop_size = max(4 * dim, 10, int(0.05 * budget))  # budget-sensitive
        pop_size = min(pop_size, budget - 1)  # ensure room for evaluations
        max_evals = budget
        evals = 0

        # Latin Hypercube Sampling for initial population
        def latin_hypercube(n, d, lb, ub, rng):
            samples = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                samples[:, j] = (perm + rng.uniform(size=n)) / n
            # Scale to bounds
            samples = lb + samples * (ub - lb)
            return samples

        pop = latin_hypercube(pop_size, dim, lb, ub, rng)
        fitness = np.full(pop_size, np.inf)
        best_x = pop[0].copy()
        best_f = np.inf

        for i in range(pop_size):
            if evals >= max_evals:
                break
            f = func(pop[i])
            evals += 1
            fitness[i] = f
            if f < best_f:
                best_f = f
                best_x = pop[i].copy()
                report_best(best_f, best_x)

        # DE parameters
        F = 0.5
        CR = 0.9

        # Stagnation detection
        stagnation_limit = max(50, int(0.1 * budget))
        evals_no_improve = 0
        last_improve_evals = 0

        while evals < max_evals:
            # Generate new population
            new_pop = np.empty_like(pop)
            new_fitness = np.empty(pop_size)
            for i in range(pop_size):
                if evals >= max_evals:
                    break
                # Select three distinct random indices other than i
                idxs = [j for j in range(pop_size) if j != i]
                rng.shuffle(idxs)
                a, b, c = idxs[:3]
                # Mutation
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Crossover with random binomial
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                # Clip to bounds
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                new_pop[i] = trial
                new_fitness[i] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
                    report_best(best_f, best_x)
                    evals_no_improve = 0
                    last_improve_evals = evals
                else:
                    # No improvement from this trial, but still use selection later
                    pass

            if evals >= max_evals:
                break

            # Selection: replace if better or equal
            for i in range(pop_size):
                if new_fitness[i] <= fitness[i]:
                    pop[i] = new_pop[i]
                    fitness[i] = new_fitness[i]

            # Check stagnation
            if evals - last_improve_evals >= stagnation_limit:
                # Restart half of population (except best)
                restart_size = pop_size // 2
                # Identify worst indices to replace
                order = np.argsort(fitness)
                worst_idxs = order[-restart_size:]
                # Generate new points with Latin Hypercube
                new_samples = latin_hypercube(restart_size, dim, lb, ub, rng)
                for idx, new_x in zip(worst_idxs, new_samples):
                    if evals >= max_evals:
                        break
                    # Ensure we don't overwrite best
                    if idx == order[0]:  # best
                        continue
                    f = func(new_x)
                    evals += 1
                    pop[idx] = new_x
                    fitness[idx] = f
                    if f < best_f:
                        best_f = f
                        best_x = new_x.copy()
                        report_best(best_f, best_x)
                        last_improve_evals = evals
                # Reset stagnation counter
                evals_no_improve = 0
                last_improve_evals = evals

        return best_f, best_x