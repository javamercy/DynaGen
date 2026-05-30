import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.evals = 0
        self.best_value = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        pop_size = max(3, min(4 * dim, budget // 2))
        pop = lb + (ub - lb) * rng.rand(pop_size, dim)
        pop_fitness = np.full(pop_size, np.inf)

        for i in range(pop_size):
            if self.evals >= budget:
                break
            x = pop[i]
            val = func(x)
            self.evals += 1
            pop_fitness[i] = val
            if val < self.best_value:
                self.best_value = val
                self.best_x = x.copy()
                report_best(val, x)

        best_idx = np.argmin(pop_fitness)
        best = pop[best_idx].copy()

        F = 0.8
        CR = 0.9

        no_improve_streak = 0

        while self.evals < budget:
            # DE generation
            for i in range(pop_size):
                if self.evals >= budget:
                    break
                candidates = [j for j in range(pop_size) if j != i]
                selected = rng.choice(candidates, 2, replace=False)
                a, b = selected
                mutant = best + F * (pop[a] - pop[b])
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                self.evals += 1
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                    if val < self.best_value:
                        self.best_value = val
                        self.best_x = trial.copy()
                        report_best(val, trial)
                        best = trial.copy()
                        no_improve_streak = 0
                else:
                    no_improve_streak += 1

            # Local search around best
            if self.evals < budget:
                local_trials = min(pop_size, budget - self.evals)
                if local_trials > 0:
                    remaining_frac = (budget - self.evals) / budget
                    sigma = 0.005 * (ub - lb) * remaining_frac
                    improved_local = False
                    for _ in range(local_trials):
                        if self.evals >= budget:
                            break
                        perturbation = rng.normal(0, sigma, size=dim)
                        trial = np.clip(best + perturbation, lb, ub)
                        val = func(trial)
                        self.evals += 1
                        if val < self.best_value:
                            self.best_value = val
                            self.best_x = trial.copy()
                            report_best(val, trial)
                            best = trial.copy()
                            improved_local = True
                    if not improved_local:
                        if no_improve_streak > pop_size * 2 and budget - self.evals >= pop_size:
                            radius = 0.1 * (ub - lb) * remaining_frac
                            for i in range(pop_size):
                                if self.evals >= budget:
                                    break
                                x = best + rng.uniform(-radius, radius, size=dim)
                                x = np.clip(x, lb, ub)
                                val = func(x)
                                self.evals += 1
                                pop[i] = x
                                pop_fitness[i] = val
                                if val < self.best_value:
                                    self.best_value = val
                                    self.best_x = x.copy()
                                    report_best(val, x)
                                    best = x.copy()
                            no_improve_streak = 0

        return self.best_value, self.best_x