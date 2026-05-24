import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.best_value = float('inf')
        self.best_x = None

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        evals = 0

        # Population size
        popsize = min(budget, max(4, min(5*dim, 30)))
        if popsize < 4:
            popsize = min(budget, 4)
        pop = np.array([rng.uniform(lb, ub, dim) for _ in range(popsize)])
        pop_fit = np.array([func(pop[i]) for i in range(popsize)])
        evals += popsize
        for i in range(popsize):
            if pop_fit[i] < self.best_value:
                self.best_value = pop_fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)

        if evals >= budget:
            return self.best_value, self.best_x

        # Success memory for adaptation
        mem_size = 5
        mem_F = np.full(mem_size, 0.6)
        mem_CR = np.full(mem_size, 0.5)
        mem_idx = 0
        c = 0.1  # smoothing factor

        stagnation_counter = 0
        max_stagnation = max(5, int(budget / (popsize * 2)))

        while evals < budget:
            success_F = []
            success_CR = []
            new_pop = np.empty_like(pop)
            new_fit = np.empty(popsize)
            for i in range(popsize):
                # Sample F and CR from memory
                k = rng.randint(mem_size)
                F = mem_F[k] + 0.1 * rng.standard_cauchy()
                F = np.clip(F, 0.1, 1.0)
                CR = mem_CR[k] + 0.1 * rng.randn()
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: DE/rand/2
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                r1, r2, r3, r4, r5 = candidates[:5]
                mutant = pop[r1] + F * (pop[r2] - pop[r3]) + F * (pop[r4] - pop[r5])

                # Crossover: binomial
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)

                # Evaluation
                trial_fit = func(trial)
                evals += 1
                if trial_fit <= pop_fit[i]:
                    new_pop[i] = trial
                    new_fit[i] = trial_fit
                    success_F.append(F)
                    success_CR.append(CR)
                    if trial_fit < self.best_value:
                        self.best_value = trial_fit
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                else:
                    new_pop[i] = pop[i]
                    new_fit[i] = pop_fit[i]

                if evals >= budget:
                    break

            if evals >= budget:
                break

            # Update population
            pop = new_pop
            pop_fit = new_fit

            # Update success memory
            if len(success_F) > 0:
                mean_F = np.mean(success_F)
                mean_CR = np.mean(success_CR)
                mem_F[mem_idx] = (1 - c) * mem_F[mem_idx] + c * mean_F
                mem_CR[mem_idx] = (1 - c) * mem_CR[mem_idx] + c * mean_CR
                mem_idx = (mem_idx + 1) % mem_size

            # Check improvement
            if np.any(new_fit < pop_fit):
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Restart if stagnation
            if stagnation_counter >= max_stagnation and evals + popsize - 1 <= budget:
                # Keep best, reinitialize rest
                new_pop = np.empty_like(pop)
                new_fit = np.empty(popsize)
                new_pop[0] = self.best_x
                new_fit[0] = self.best_value
                for i in range(1, popsize):
                    x = rng.uniform(lb, ub, dim)
                    f = func(x)
                    evals += 1
                    new_pop[i] = x
                    new_fit[i] = f
                    if f < self.best_value:
                        self.best_value = f
                        self.best_x = x.copy()
                        report_best(self.best_value, self.best_x)
                pop = new_pop
                pop_fit = new_fit
                stagnation_counter = 0
                if evals >= budget:
                    break

        return self.best_value, self.best_x