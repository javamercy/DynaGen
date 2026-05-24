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
        popsize = min(budget, max(10, 5*dim))
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

        F1 = 0.8
        F2 = 0.9
        CR = 0.9
        stagnation_counter = 0
        max_stagnation = max(5, int(budget / (3 * popsize)))

        smoothed_success = np.array([0.5, 0.5])
        beta = 0.1
        epsilon = 1e-6

        local_search_scale = 0.05 * (ub - lb) / np.sqrt(dim)

        while evals < budget:
            improved_this_gen = False
            num_improved = np.zeros(2, dtype=int)
            num_attempts = np.zeros(2, dtype=int)

            for i in range(popsize):
                p = smoothed_success[0] / (smoothed_success[0] + smoothed_success[1] + epsilon)
                if rng.rand() < p:
                    strategy = 0
                    candidates = list(range(popsize))
                    candidates.remove(i)
                    rng.shuffle(candidates)
                    r1, r2 = candidates[0], candidates[1]
                    mutant = pop[i] + F1 * (self.best_x - pop[i]) + F1 * (pop[r1] - pop[r2])
                else:
                    strategy = 1
                    candidates = list(range(popsize))
                    candidates.remove(i)
                    rng.shuffle(candidates)
                    r1, r2, r3 = candidates[0], candidates[1], candidates[2]
                    mutant = pop[r1] + F2 * (pop[r2] - pop[r3])

                trial = np.copy(pop[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_fitness = func(trial)
                evals += 1
                num_attempts[strategy] += 1
                if trial_fitness <= pop_fitness[i]:
                    pop_fitness[i] = trial_fitness
                    pop[i] = trial
                    num_improved[strategy] += 1
                    if trial_fitness < self.best_value:
                        self.best_value = trial_fitness
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        improved_this_gen = True
                if evals >= budget:
                    break

            if evals >= budget:
                break

            for s in range(2):
                if num_attempts[s] > 0:
                    current_rate = num_improved[s] / num_attempts[s]
                    smoothed_success[s] = (1 - beta) * smoothed_success[s] + beta * current_rate
            smoothed_success = np.maximum(smoothed_success, epsilon)

            # Local search around best point (exploitation)
            if improved_this_gen and evals < budget:
                local_evals = min(5, budget - evals)
                for _ in range(local_evals):
                    candidate = self.best_x + local_search_scale * rng.randn(dim)
                    candidate = np.clip(candidate, lb, ub)
                    f = func(candidate)
                    evals += 1
                    if f < self.best_value:
                        self.best_value = f
                        self.best_x = candidate.copy()
                        report_best(self.best_value, self.best_x)
                    if evals >= budget:
                        break

            if evals >= budget:
                break

            if improved_this_gen:
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter >= max_stagnation and evals + popsize - 1 <= budget:
                # Covariance-adapted restart (exploitation-focused: smaller scale)
                n_best = max(2, int(popsize / 3))
                sorted_indices = np.argsort(pop_fitness)
                best_indices = sorted_indices[:n_best]
                best_pop = pop[best_indices]
                if len(best_pop) >= 2:
                    cov = np.cov(best_pop, rowvar=False)
                else:
                    cov = np.eye(dim) * 1e-6
                try:
                    L = np.linalg.cholesky(cov)
                except np.linalg.LinAlgError:
                    cov += np.eye(dim) * 1e-6
                    L = np.linalg.cholesky(cov)
                scale = 0.1 / dim * (ub - lb)
                new_pop = np.zeros((popsize, dim))
                new_fitness = np.zeros(popsize)
                new_pop[0] = self.best_x
                new_fitness[0] = self.best_value
                for i in range(1, popsize):
                    sample = self.best_x + scale * (L @ rng.randn(dim))
                    sample = np.clip(sample, lb, ub)
                    new_pop[i] = sample
                    new_fitness[i] = func(sample)
                    evals += 1
                    if new_fitness[i] < self.best_value:
                        self.best_value = new_fitness[i]
                        self.best_x = new_pop[i].copy()
                        report_best(self.best_value, self.best_x)
                pop = new_pop
                pop_fitness = new_fitness
                stagnation_counter = 0
                smoothed_success[:] = 0.5
                if evals >= budget:
                    break

        return self.best_value, self.best_x