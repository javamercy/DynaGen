import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.best_value = float('inf')
        self.best_x = None

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget must be positive")
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        evals = 0
        CR = 0.9
        popsize = min(budget, max(4 * dim, 20))
        # Ensure at least 4 individuals for DE/rand/1
        if popsize < 4:
            popsize = min(budget, 4)

        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        pop_fitness = np.full(popsize, float('inf'))
        for i in range(popsize):
            pop_fitness[i] = func(pop[i])
            evals += 1
            if pop_fitness[i] < self.best_value:
                self.best_value = pop_fitness[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x

        stagnation = 0
        prev_best = self.best_value

        while evals < budget:
            improved = False
            for i in range(popsize):
                if evals >= budget:
                    break
                # select three distinct indices different from i
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]
                F = rng.uniform(0.5, 1.0)
                mutant = pop[a] + F * (pop[b] - pop[c])
                # binomial crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
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
                        improved = True

            if evals >= budget:
                break

            if improved:
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= dim:
                # restart worst half with best-perturbed reinitialization
                order = np.argsort(pop_fitness)[::-1]  # worst first
                num_restart = max(1, popsize // 2)
                for idx in order[:num_restart]:
                    if evals >= budget:
                        break
                    # best-perturbed
                    scale = 0.2 * (ub - lb)
                    new_x = self.best_x + scale * rng.randn(dim)
                    new_x = np.clip(new_x, lb, ub)
                    new_fitness = func(new_x)
                    evals += 1
                    if new_fitness < self.best_value:
                        self.best_value = new_fitness
                        self.best_x = new_x.copy()
                        report_best(self.best_value, self.best_x)
                    pop[idx] = new_x
                    pop_fitness[idx] = new_fitness
                stagnation = 0

        return self.best_value, self.best_x