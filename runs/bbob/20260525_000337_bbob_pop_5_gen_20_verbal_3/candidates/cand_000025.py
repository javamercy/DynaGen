import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        rng = self.rng
        lb = np.asarray(func.bounds.lb)
        ub = np.asarray(func.bounds.ub)
        # Population size
        pop_size = min(budget, max(10, 4 * dim))
        # Initial population
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        fit = np.full(pop_size, np.inf)
        best_x = None
        best_value = np.inf
        evals = 0
        for i in range(pop_size):
            val = func(pop[i])
            evals += 1
            fit[i] = val
            if val < best_value:
                best_value = val
                best_x = pop[i].copy()
                report_best(best_value, best_x)
        # DE parameters
        F = 0.8
        Cr = 0.9
        # Stagnation tracking
        evals_since_improve = 0
        restart_threshold = int(0.2 * budget)
        # Main loop
        while evals < budget:
            # Check restart condition
            if evals_since_improve >= restart_threshold:
                # Restart: keep best, reinitialize rest uniformly
                if best_x is not None:
                    new_pop = np.empty((pop_size, dim))
                    new_pop[0] = best_x.copy()
                    for i in range(1, pop_size):
                        new_pop[i] = rng.uniform(lb, ub)
                    pop = new_pop
                else:
                    pop = rng.uniform(lb, ub, size=(pop_size, dim))
                # Reevaluate all
                for i in range(pop_size):
                    val = func(pop[i])
                    evals += 1
                    fit[i] = val
                    if val < best_value:
                        best_value = val
                        best_x = pop[i].copy()
                        report_best(best_value, best_x)
                evals_since_improve = 0
                if evals >= budget:
                    break
            # Determine how many trials to generate this generation
            gen_trials = min(pop_size, budget - evals)
            # For each target individual
            for i in range(gen_trials):
                # Mutation: select three distinct random indices != i
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]
                # Mutant vector
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Binomial crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < Cr or j == j_rand:
                        trial[j] = mutant[j]
                # Clip to bounds
                trial = np.clip(trial, lb, ub)
                # Evaluate trial
                val = func(trial)
                evals += 1
                if val < fit[i]:
                    pop[i] = trial
                    fit[i] = val
                    if val < best_value:
                        best_value = val
                        best_x = trial.copy()
                        evals_since_improve = 0
                        report_best(best_value, best_x)
                        # Early exit if we just found best and budget exhausted
                        if evals >= budget:
                            break
                else:
                    evals_since_improve += 1
            if evals >= budget:
                break
        return best_value, best_x