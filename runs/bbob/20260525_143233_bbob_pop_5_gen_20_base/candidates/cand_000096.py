import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(5 * dim, budget // 2))
        self.restart_threshold = max(10, 2 * dim)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng
        pop_size = self.pop_size

        if budget == 0:
            best_x = (lb + ub) / 2.0
            best_val = func(best_x)
            report_best(best_val, best_x)
            return best_val, best_x

        if pop_size < 2 or budget < 2:
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
        # Self-adaptive parameters: F and CR per individual
        F = rng.uniform(0.5, 1.0, pop_size)
        CR = rng.uniform(0.0, 1.0, pop_size)

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

        no_improve = 0
        tau1 = 0.1
        tau2 = 0.1

        while evals < budget:
            improved = False
            new_pop = pop.copy()
            new_F = F.copy()
            new_CR = CR.copy()
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Generate new F and CR with probabilities
                if rng.rand() < tau1:
                    new_F[i] = rng.uniform(0.1, 1.0)
                if rng.rand() < tau2:
                    new_CR[i] = rng.uniform(0.0, 1.0)
                # Mutation: rand/1
                candidates = [j for j in range(pop_size) if j != i]
                if len(candidates) < 3:
                    continue
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                F_i = new_F[i]
                mutant = pop[r1] + F_i * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)
                # Crossover
                cross_points = rng.rand(dim) < new_CR[i]
                if not np.any(cross_points):
                    cross_points[rng.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = val
                    improved = True
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                    # Keep the new parameters that led to success
                    # (already assigned to new_F, new_CR)
                else:
                    # Revert parameters to old ones
                    new_F[i] = F[i]
                    new_CR[i] = CR[i]
            pop = new_pop
            F = new_F
            CR = new_CR

            if improved:
                no_improve = 0
            else:
                no_improve += 1

            # Restart if stagnation
            if no_improve >= self.restart_threshold and evals < budget:
                # Reinitialize population except best
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
                # Reinitialize F and CR
                F = rng.uniform(0.5, 1.0, pop_size)
                CR = rng.uniform(0.0, 1.0, pop_size)
                no_improve = 0

        return best_val, best_x