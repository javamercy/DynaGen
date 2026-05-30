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

        # Population size
        pop_size = max(4 * dim, 20)
        pop_size = min(pop_size, budget // 2)
        if pop_size < 5:
            pop_size = min(5, budget)

        # Latin Hypercube Sampling
        def lhs(n, d):
            intervals = np.linspace(0, 1, n + 1)
            samples = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                for i in range(n):
                    samples[i, j] = intervals[perm[i]] + rng.uniform(0, 1/n)
            return samples

        pop = lb + (ub - lb) * lhs(pop_size, dim)
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        # Evaluate initial population
        for i in range(pop_size):
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            if evals >= budget:
                return best_val, best_x

        # DE parameters (fixed)
        F = 0.8
        CR = 0.9

        # Stagnation parameters
        stag_limit = max(20, pop_size)
        stag_counter = 0
        last_best = best_val

        max_gen = (budget - evals) // pop_size
        gen = 0
        while evals < budget and gen < max_gen:
            # Mutation and crossover
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Select three distinct random indices
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]

                # Mutant
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Binomial crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # Stagnation check
            if best_val < last_best:
                stag_counter = 0
                last_best = best_val
            else:
                stag_counter += 1

            if stag_counter >= stag_limit and evals < budget:
                # Replace worst 30% with random points
                n_replace = max(1, pop_size // 3)
                worst_idx = np.argsort(fitness)[-n_replace:]
                for idx in worst_idx:
                    if evals >= budget:
                        break
                    new_x = lb + rng.rand(dim) * (ub - lb)
                    new_val = func(new_x)
                    evals += 1
                    pop[idx] = new_x
                    fitness[idx] = new_val
                    if new_val < best_val:
                        best_val = new_val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)
                stag_counter = 0
                last_best = best_val

            gen += 1

        return best_val, best_x