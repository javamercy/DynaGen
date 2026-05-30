import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Population size: max(5, 4*dim) but capped at 30 and at least 2
        pop_size = max(2, min(4*dim, 30, budget // 2))
        if pop_size < 2:
            pop_size = 2  # ensure at least 2 for DE mutation

        # Initialize population
        pop = rng.uniform(lb, ub, (pop_size, dim))
        fit = np.full(pop_size, np.inf)
        best_x = pop[0].copy()
        best_f = np.inf
        calls = 0

        for i in range(pop_size):
            if calls >= budget:
                break
            fit[i] = func(pop[i])
            calls += 1
            if fit[i] < best_f:
                best_f = fit[i]
                best_x = pop[i].copy()
                report_best(best_f, best_x)

        # DE parameters
        F = 0.9
        CR = 0.9
        stagnation_generations = 0
        max_stagnation = 5  # restart after 5 gens without improvement
        best_f_prev = best_f

        while calls < budget:
            improved = False
            # Generate trial vectors for each target
            for i in range(pop_size):
                if calls >= budget:
                    break
                # Select distinct a,b,c different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]

                # Mutation and crossover
                mut = pop[a] + F * (pop[b] - pop[c])
                # Binomial crossover
                trial = np.where(rng.rand(dim) < CR, mut, pop[i])
                # Ensure at least one component from mut
                j_rand = rng.randint(dim)
                trial[j_rand] = mut[j_rand]
                # Clip to bounds
                trial = np.clip(trial, lb, ub)

                # Evaluate
                ftrial = func(trial)
                calls += 1
                if ftrial < fit[i]:
                    pop[i] = trial.copy()
                    fit[i] = ftrial
                    if ftrial < best_f:
                        best_f = ftrial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        improved = True

            # Stagnation detection and restart
            if not improved:
                stagnation_generations += 1
            else:
                stagnation_generations = 0

            if stagnation_generations >= max_stagnation:
                # Reinitialize population except best
                # Keep best individual
                for i in range(pop_size):
                    if fit[i] > best_f:  # only replace worse ones
                        pop[i] = rng.uniform(lb, ub, dim)
                        if calls >= budget:
                            break
                        fit[i] = func(pop[i])
                        calls += 1
                        if fit[i] < best_f:
                            best_f = fit[i]
                            best_x = pop[i].copy()
                            report_best(best_f, best_x)
                stagnation_generations = 0

        return best_f, best_x