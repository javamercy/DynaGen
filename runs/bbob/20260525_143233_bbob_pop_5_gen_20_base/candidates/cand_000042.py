import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        # population size: at least 4, scaled by sqrt(dim), capped by budget
        self.pop_size = max(4, min(4 * int(np.sqrt(dim)), budget // 2))
        self.pop_size = min(self.pop_size, budget)
        self.restart_threshold = max(5, 2 * dim)
        # memory for successful F and CR (JADE style)
        self.memory_size = 5
        self.mu_F = 0.5
        self.mu_CR = 0.9
        self.archive_F = [0.5] * self.memory_size
        self.archive_CR = [0.9] * self.memory_size
        self.memory_counter = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        pop_size = self.pop_size
        rng = self.rng

        # fallback for tiny budget
        if pop_size <= 2:
            best_x = rng.uniform(lb, ub, dim)
            best_val = func(best_x)
            report_best(best_val, best_x)
            evals = 1
            while evals < budget:
                x = rng.uniform(lb, ub, dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        # initialization
        pop = rng.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0
        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i].copy()
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        no_improve = 0
        generation = 0
        successful_F = []
        successful_CR = []

        while evals < budget:
            improved_this_gen = False
            successful_F.clear()
            successful_CR.clear()
            # sample F and CR for each individual from Cauchy distributions
            # F ~ Cauchy(mu_F, 0.1) truncated to [0, 1]
            # CR ~ Cauchy(mu_CR, 0.1) truncated to [0, 1]
            F_samples = np.clip(rng.standard_cauchy(pop_size) * 0.1 + self.mu_F, 0, 1)
            CR_samples = np.clip(rng.standard_cauchy(pop_size) * 0.1 + self.mu_CR, 0, 1)
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 2:
                    continue
                r1, r2 = rng.choice(candidates, size=2, replace=False)
                # current-to-best/1
                F = F_samples[i]
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                CR = CR_samples[i]
                cross_points = rng.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    improved_this_gen = True
                    successful_F.append(F)
                    successful_CR.append(CR)
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # update memory with Lehmer mean of successful parameters
            if improved_this_gen:
                no_improve = 0
                if len(successful_F) > 0:
                    # Lehmer mean
                    F_lehmer = np.sum(successful_F**2) / np.sum(successful_F)
                    CR_lehmer = np.sum(successful_CR**2) / np.sum(successful_CR)
                    # replace oldest entry in memory
                    self.archive_F[self.memory_counter % self.memory_size] = F_lehmer
                    self.archive_CR[self.memory_counter % self.memory_size] = CR_lehmer
                    self.memory_counter += 1
                    self.mu_F = np.mean(self.archive_F)
                    self.mu_CR = np.mean(self.archive_CR)
            else:
                no_improve += 1

            # local search around best
            if evals < budget:
                local_evals = min(2, budget - evals)
                sigma = 0.01 * (ub - lb)
                for _ in range(local_evals):
                    x = best_x + sigma * rng.randn(dim)
                    x = np.clip(x, lb, ub)
                    val = func(x)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)

            # restart if stagnation
            if no_improve >= self.restart_threshold:
                new_pop = rng.uniform(lb, ub, (pop_size, dim))
                if best_x is not None:
                    focused_count = max(1, int(0.3 * pop_size))
                    for j in range(focused_count):
                        sigma = 0.1 * (ub - lb)
                        x = best_x + sigma * rng.randn(dim)
                        x = np.clip(x, lb, ub)
                        new_pop[j] = x
                    new_pop[0] = best_x.copy()
                new_fitness = np.full(pop_size, np.inf)
                new_fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    x = new_pop[i].copy()
                    val = func(x)
                    evals += 1
                    new_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                pop = new_pop
                fitness = new_fitness
                self.mu_F = 0.5
                self.mu_CR = 0.9
                self.archive_F = [0.5] * self.memory_size
                self.archive_CR = [0.9] * self.memory_size
                self.memory_counter = 0
                no_improve = 0

            generation += 1

        return best_val, best_x