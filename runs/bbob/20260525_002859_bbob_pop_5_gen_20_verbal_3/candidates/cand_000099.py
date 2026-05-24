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
        dim = self.dim
        budget = self.budget
        lb = func.bounds.lb
        ub = func.bounds.ub

        popsize = max(4, min(5*dim, budget//10, 30))
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        fit = np.full(popsize, np.inf)
        evals = 0

        # initial evaluation
        for i in range(popsize):
            if evals >= budget:
                break
            fit[i] = func(pop[i])
            evals += 1
            if i == 0 or fit[i] < self.best_value:
                self.best_value = fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x

        # DE parameters
        F = 0.8
        CR = 0.9
        success_F = []
        success_CR = []
        max_memory = 20
        stagnation_counter = 0
        stagnation_limit = max(10 * dim, int(0.2 * budget))
        prob_rand1 = 0.2

        while evals < budget:
            improved = False
            for i in range(popsize):
                if evals >= budget:
                    break
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                r1, r2, r3 = candidates[0], candidates[1], candidates[2]
                if rng.rand() < prob_rand1:
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])
                else:
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
                    if trial_fit < self.best_value:
                        self.best_value = trial_fit
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        stagnation_counter = 0
                        improved = True
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1

            if evals >= budget:
                break

            if improved:
                success_F.append(F)
                success_CR.append(CR)
                if len(success_F) > max_memory:
                    success_F.pop(0)
                    success_CR.pop(0)
                F = np.mean(success_F)
                CR = np.mean(success_CR)

            # restart: replace all individuals except best with random points
            if stagnation_counter >= stagnation_limit and evals < budget - popsize + 1:
                stagnation_counter = 0
                # keep best
                new_pop = [self.best_x.copy()]
                for _ in range(popsize - 1):
                    new_x = lb + (ub - lb) * rng.rand(dim)
                    new_pop.append(new_x)
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
                pop[0] = self.best_x.copy()
                fit[0] = self.best_value

        return self.best_value, self.best_x