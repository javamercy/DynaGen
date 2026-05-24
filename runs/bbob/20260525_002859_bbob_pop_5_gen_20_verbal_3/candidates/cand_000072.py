import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.best_value = None
        self.best_x = None

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        evals = 0

        # population size
        popsize = min(budget, max(4, min(5*dim, 20)))
        if popsize < 4:
            popsize = 4

        # initial population
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        fit = np.full(popsize, np.inf)
        for i in range(popsize):
            fit[i] = func(pop[i])
            evals += 1
            if evals == 1 or fit[i] < self.best_value:
                self.best_value = fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)

        if evals >= budget:
            return self.best_value, self.best_x

        # parameter memory
        memory_size = 10
        F_memory = [0.8] * memory_size
        CR_memory = [0.8] * memory_size
        memory_idx = 0
        F = 0.8
        CR = 0.9

        # stagnation detection
        stagn_gen = 0
        max_stagn_gen = max(10, int(budget / (popsize * 5)))
        diversity_threshold = 0.05 * (ub - lb).mean()  # relative diversity

        while evals < budget:
            successful_F = []
            successful_CR = []
            best_idx = np.argmin(fit)
            self.best_x = pop[best_idx].copy()
            self.best_value = fit[best_idx]

            for i in range(popsize):
                if evals >= budget:
                    break
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                r1, r2 = candidates[0], candidates[1]
                mutant = pop[i] + F * (self.best_x - pop[i]) + F * (pop[r1] - pop[r2])
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_fit = func(trial)
                evals += 1
                if trial_fit <= fit[i]:
                    fit[i] = trial_fit
                    pop[i] = trial
                    successful_F.append(F)
                    successful_CR.append(CR)
                    if trial_fit < self.best_value:
                        self.best_value = trial_fit
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)

            # diversity measure: mean of per-dimension std
            if popsize > 1:
                diversity = np.mean(np.std(pop, axis=0))
            else:
                diversity = 0.0

            # stagnation check
            best_improved = np.min(fit) < self.best_value - 1e-12
            if best_improved:
                stagn_gen = 0
                self.best_value = np.min(fit)
                self.best_x = pop[np.argmin(fit)].copy()
            else:
                stagn_gen += 1

            # update adaptive parameters
            if len(successful_F) > 0:
                F_new = np.sum(np.array(successful_F)**2) / np.sum(successful_F) if np.sum(successful_F) > 0 else 0.8
                CR_new = np.mean(successful_CR)
                F_memory[memory_idx] = min(F_new, 0.9)
                CR_memory[memory_idx] = min(CR_new, 1.0)
                memory_idx = (memory_idx + 1) % memory_size
                F = np.mean(F_memory)
                CR = np.mean(CR_memory)

            # restart condition: stagnation and low diversity
            restart_condition = (stagn_gen >= max_stagn_gen) and (diversity < diversity_threshold) and (evals < budget - popsize)
            if restart_condition:
                # keep best
                new_pop = [self.best_x.copy()]
                # generate others: mix of Cauchy around best and uniform
                for _ in range(popsize - 1):
                    if rng.rand() < 0.5:
                        # Cauchy perturbation from best
                        sigma = (ub - lb) * 0.2 * (1 + rng.rand())
                        new_x = self.best_x + sigma * rng.standard_cauchy(dim)
                        new_x = np.clip(new_x, lb, ub)
                    else:
                        new_x = lb + (ub - lb) * rng.rand(dim)
                    new_pop.append(new_x)
                # evaluate new individuals (skip already evaluated best)
                for j in range(1, popsize):
                    if evals >= budget:
                        break
                    x = new_pop[j]
                    fit_val = func(x)
                    evals += 1
                    pop[j] = x
                    fit[j] = fit_val
                    if fit_val < self.best_value:
                        self.best_value = fit_val
                        self.best_x = x.copy()
                        report_best(self.best_value, self.best_x)
                # reset stagnation
                stagn_gen = 0
                # reset F and CR to memory means (which persist)
                F = np.mean(F_memory)
                CR = np.mean(CR_memory)

            if evals >= budget:
                break

        return self.best_value, self.best_x