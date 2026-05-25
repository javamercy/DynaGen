import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(5 * dim, budget // 4))
        # Initialize F and CR for each individual
        self.F = None
        self.CR = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget
        rng = self.rng

        if pop_size < 2:
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

        # Initialize population
        pop = rng.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0
        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # Initialize self-adaptive parameters
        self.F = np.full(pop_size, 0.5)
        self.CR = np.full(pop_size, 0.9)
        # Track success for adaptation
        success_F = []
        success_CR = []

        no_improve = 0
        restart_threshold = max(10, 2 * dim)
        # Initial diversity measure (variance)
        initial_var = np.mean(np.var(pop, axis=0))
        diversity_threshold = max(1e-12, 0.01 * initial_var)  # relative threshold

        while evals < budget:
            # Local search every 5 generations
            if no_improve % 5 == 0 and evals < budget:
                for _ in range(min(5, budget - evals)):
                    sigma = 0.02 * (ub - lb) * (1 - evals / budget)
                    trial = best_x + sigma * rng.randn(dim)
                    trial = np.clip(trial, lb, ub)
                    val = func(trial)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        no_improve = 0

            improved = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Select two distinct random indices
                candidates = [j for j in range(pop_size) if j != i]
                if len(candidates) < 2:
                    continue
                r1, r2 = rng.choice(candidates, size=2, replace=False)
                F_i = self.F[i]
                CR_i = self.CR[i]
                mutant = pop[i] + F_i * (best_x - pop[i]) + F_i * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover
                cross_points = rng.rand(dim) < CR_i
                if not np.any(cross_points):
                    cross_points[rng.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    improved = True
                    success_F.append(F_i)
                    success_CR.append(CR_i)
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        no_improve = 0

            # Adapt parameters using success history
            if len(success_F) > 0:
                # Update each individual's F and CR based on successful values
                # Here we set all to the mean of successful ones (like jDE but simpler)
                mean_F = np.mean(success_F)
                mean_CR = np.mean(success_CR)
                self.F = np.full(pop_size, mean_F)
                self.CR = np.full(pop_size, mean_CR)
                success_F = []
                success_CR = []
            else:
                # Increase exploration
                self.F = np.clip(self.F * 1.1, 0.1, 1.0)
                self.CR = np.clip(self.CR * 0.95, 0.1, 1.0)

            if improved:
                no_improve = 0
            else:
                no_improve += 1

            # Diversity check for restart
            current_var = np.mean(np.var(pop, axis=0))
            if current_var < diversity_threshold and no_improve >= 5:
                # Restart
                new_pop = rng.uniform(lb, ub, (pop_size, dim))
                new_pop[0] = best_x.copy()
                new_fitness = np.full(pop_size, np.inf)
                new_fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    x = new_pop[i]
                    val = func(x)
                    evals += 1
                    new_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                pop = new_pop
                fitness = new_fitness
                # Reset parameters to default
                self.F = np.full(pop_size, 0.5)
                self.CR = np.full(pop_size, 0.9)
                no_improve = 0
                # Update diversity threshold
                initial_var = np.mean(np.var(pop, axis=0))
                diversity_threshold = max(1e-12, 0.01 * initial_var)

        return best_val, best_x