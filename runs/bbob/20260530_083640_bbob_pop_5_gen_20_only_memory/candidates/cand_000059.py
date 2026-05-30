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

        # population size
        pop_size = min(10 * dim, budget // 3)
        pop_size = max(pop_size, 5)

        # initial population
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        pop_vals = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        # evaluate initial
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

        # track stagnation
        last_improvement_gen = 0
        gen = 0

        while evals < budget:
            # adaptive parameters
            F = 0.5 + 0.3 * rng.rand()
            CR = 0.1 + 0.8 * rng.rand()

            new_pop = pop.copy()
            for i in range(pop_size):
                if evals >= budget:
                    break
                # mutation
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(idxs, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                # cauchy mutation occasionally for diversity
                if rng.rand() < 0.1:
                    mutant += 0.1 * rng.standard_cauchy(dim)
                # crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < pop_vals[i]:
                    new_pop[i] = trial
                    pop_vals[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        last_improvement_gen = gen

            pop = new_pop

            # restart worst if stagnation
            if gen - last_improvement_gen >= 5 and evals < budget:
                # reinitialize worst 20%
                n_restart = max(1, pop_size // 5)
                worst_idx = np.argsort(pop_vals)[-n_restart:]
                for idx in worst_idx:
                    if evals >= budget:
                        break
                    pop[idx] = rng.uniform(lb, ub)
                    val = func(pop[idx])
                    evals += 1
                    pop_vals[idx] = val
                    if val < best_val:
                        best_val = val
                        best_x = pop[idx].copy()
                        report_best(best_val, best_x)
                last_improvement_gen = gen

            gen += 1

        return best_val, best_x