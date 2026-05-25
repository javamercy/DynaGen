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

        # Population size
        NP = min(10 * dim, max(4, budget // 2 - 1))
        if NP < 4:
            NP = 4
        if budget < NP:
            NP = budget

        # Initialize
        pop = lb + (ub - lb) * rng.rand(NP, dim)
        fitness = np.asarray([func(pop[i]) for i in range(NP)])
        func_evals = NP

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_val = fitness[best_idx]
        report_best(best_val, best_x)

        # Success-history memory
        M = 10
        hist_F = np.full(M, 0.8)
        hist_CR = np.full(M, 0.9)
        hist_idx = 0

        stagnation_counter = 0
        max_stagnation = 10
        diversity_threshold = 1e-6 * np.mean(ub - lb)
        max_restarts = 2
        restart_count = 0

        while func_evals < budget:
            # generate offspring
            for i in range(NP):
                if func_evals >= budget:
                    break
                # random indices
                candidates = list(range(NP))
                candidates.remove(i)
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                # sample parameters
                k = rng.randint(M)
                F = rng.standard_cauchy() * 0.1 + hist_F[k]
                F = np.clip(F, 0, 1)
                CR = rng.normal(hist_CR[k], 0.1)
                CR = np.clip(CR, 0, 1)
                # mutant
                a, b, c = pop[r1], pop[r2], pop[r3]
                mutant = a + F * (b - c)
                # binomial crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                # evaluate
                trial_fitness = func(trial)
                func_evals += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    # update memory
                    hist_F[hist_idx] = F
                    hist_CR[hist_idx] = CR
                    hist_idx = (hist_idx + 1) % M
                    # update best
                    if trial_fitness < best_val:
                        best_val = trial_fitness
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        stagnation_counter = 0
                else:
                    stagnation_counter += 1

            # check diversity and stagnation
            mean_std = np.mean(np.std(pop, axis=0))
            if (mean_std < diversity_threshold or stagnation_counter > max_stagnation) and \
               restart_count < max_restarts and func_evals + NP <= budget:
                # restart: keep best
                best_fitness = best_val
                best_point = best_x.copy()
                # reinitialize population
                pop = lb + (ub - lb) * rng.rand(NP, dim)
                if NP > 0:
                    pop[0] = best_point
                    fitness_eval = func(pop[i]) for i in range(NP)
                    # but we already evaluated best before; to save budget, we can assign best fitness manually
                    # Actually better: evaluate only new individuals
                    # We'll set pop[0] = best_point and fitness[0] = best_fitness
                    # Others need evaluation
                    fitness = np.empty(NP)
                    fitness[0] = best_fitness
                    for i in range(1, NP):
                        fitness[i] = func(pop[i])
                        func_evals += 1
                else:
                    # NP=0 degenerate, not possible
                    pass
                # Reset memory
                hist_F[:] = 0.8
                hist_CR[:] = 0.9
                hist_idx = 0
                stagnation_counter = 0
                restart_count += 1
                # report best again? Not needed unless improvement

            # if no restarts left and stagnation, just break or continue?
            if stagnation_counter > 10 * max_stagnation:
                break

        return best_val, best_x