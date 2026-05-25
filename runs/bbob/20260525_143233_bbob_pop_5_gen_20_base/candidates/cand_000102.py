import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(5 * dim, budget // 4))
        self.restart_threshold = max(10, 2 * dim)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget
        rng = self.rng

        if budget == 0:
            best_x = (lb + ub) / 2.0
            best_val = func(best_x)
            report_best(best_val, best_x)
            return best_val, best_x

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

        # Initialize F and CR arrays
        F = rng.uniform(0.5, 1.0, pop_size)
        CR = rng.uniform(0.0, 1.0, pop_size)

        no_improve = 0

        while evals < budget:
            improved = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Select two distinct random indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                if len(candidates) < 2:
                    continue
                r1, r2 = rng.choice(candidates, size=2, replace=False)
                # Current-to-best/1 mutation with individual F
                mutant = pop[i] + F[i] * (best_x - pop[i]) + F[i] * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover with individual CR
                cross_points = rng.rand(dim) < CR[i]
                if not np.any(cross_points):
                    cross_points[rng.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    improved = True
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                    # Keep F and CR that produced successful trial
                    # (already stored in F[i], CR[i])
                else:
                    # With probability 0.1, reinitialize F and CR
                    if rng.rand() < 0.1:
                        F[i] = rng.uniform(0.5, 1.0)
                        CR[i] = rng.uniform(0.0, 1.0)

            if improved:
                no_improve = 0
            else:
                no_improve += 1

            # Restart if stagnation
            if no_improve >= self.restart_threshold and evals < budget:
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
                # Reinitialize F and CR for new population
                F = rng.uniform(0.5, 1.0, pop_size)
                CR = rng.uniform(0.0, 1.0, pop_size)
                no_improve = 0

        return best_val, best_x