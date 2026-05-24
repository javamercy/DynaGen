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
        popsize = min(budget, max(4, min(5*dim, 20)))
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        pop_fitness = np.zeros(popsize)
        for i in range(popsize):
            pop_fitness[i] = func(pop[i])
            evals += 1
            if evals == 1 or pop_fitness[i] < self.best_value:
                self.best_value = pop_fitness[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x
        best_idx = np.argmin(pop_fitness)
        best_fitness = pop_fitness[best_idx]
        no_improve_evals = 0
        restart_evals = 0.5 * budget
        while evals < budget:
            # generate F and CR for each individual
            F = rng.cauchy(0.5, 0.3, size=popsize)
            F = np.clip(F, 0, 1)
            CR = rng.normal(0.9, 0.1, size=popsize)
            CR = np.clip(CR, 0, 1)
            for i in range(popsize):
                if evals >= budget:
                    break
                # selection of two distinct random indices different from i and best_idx
                candidates = list(range(popsize))
                candidates.remove(i)
                if best_idx in candidates:
                    candidates.remove(best_idx)
                if len(candidates) >= 2:
                    rng.shuffle(candidates)
                    a, b = candidates[:2]
                else:
                    continue
                mutant = pop[i] + F[i] * (pop[best_idx] - pop[i]) + F[i] * (pop[a] - pop[b])
                trial = np.copy(pop[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness <= pop_fitness[i]:
                    pop_fitness[i] = trial_fitness
                    pop[i] = trial
                    if trial_fitness < self.best_value:
                        self.best_value = trial_fitness
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        no_improve_evals = 0
                        best_idx = np.argmin(pop_fitness)
                        best_fitness = pop_fitness[best_idx]
                    else:
                        no_improve_evals += 1
                else:
                    no_improve_evals += 1
            if evals >= budget:
                break
            # check for restart condition
            std_fitness = np.std(pop_fitness)
            if std_fitness < 1e-4 * (abs(best_fitness) + 1e-10) or no_improve_evals >= restart_evals:
                # keep best individual, reinitialize others
                new_pop = [pop[best_idx].copy()]
                new_pop_fitness = [pop_fitness[best_idx]]
                for _ in range(1, popsize):
                    # perturb best with random scaling
                    scale = 0.1 * (ub - lb)
                    new_ind = pop[best_idx] + scale * rng.randn(dim)
                    new_ind = np.clip(new_ind, lb, ub)
                    new_pop.append(new_ind)
                    f = func(new_ind)
                    evals += 1
                    new_pop_fitness.append(f)
                    if f < self.best_value:
                        self.best_value = f
                        self.best_x = new_ind.copy()
                        report_best(self.best_value, self.best_x)
                pop = np.array(new_pop)
                pop_fitness = np.array(new_pop_fitness)
                best_idx = np.argmin(pop_fitness)
                best_fitness = pop_fitness[best_idx]
                no_improve_evals = 0
                if evals >= budget:
                    break
            # update best_idx
            best_idx = np.argmin(pop_fitness)
            best_fitness = pop_fitness[best_idx]
        return self.best_value, self.best_x